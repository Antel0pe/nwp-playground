// step_rk2.ts
import type { SimDims, FieldBuffers, Params } from "./init_boussinesq";
import type { ComputeRhsArtifacts } from "./compute_rhs";
import { makeProjection } from "./projection/project_velocity";
import { makeMicrophysicsSaturationAdjust } from "./microphysics_saturation";
import { makeDivergenceDebugger } from "./projection/debug_divergence";
import { makeFftProjectVelocity } from "./projection/fft_project_velocity";
import { makeProjectionFD4 } from "./projection/project_velocity_fd4";

// small WGSL utils (unchanged)
const CLEAR_WGSL = /*wgsl*/`
struct Uniforms { N: u32, _pad: vec3<u32>, };
@group(0) @binding(0) var<storage, read_write> buf : array<f32>;
@group(0) @binding(1) var<uniform> U : Uniforms;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }
  buf[i] = 0.0;
}
`;

const COPY_WGSL = /*wgsl*/`
struct CopyUniforms { N: u32, _pad: vec3<u32>, };
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             U   : CopyUniforms;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }
  dst[i] = src[i];
}
`;

const AXPY_WGSL = /*wgsl*/`
struct AxpyUniforms { N: u32, alpha: f32, _pad: vec2<f32>, };
@group(0) @binding(0) var<storage, read>       x   : array<f32>;
@group(0) @binding(1) var<storage, read>       y   : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<uniform>             U   : AxpyUniforms;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }
  out[i] = x[i] + U.alpha * y[i];
}
`;



type RK2Ctx = { step: (encoder: GPUCommandEncoder, dt: number) => void; };

export function makeStepRK2(opts: {
    device: GPUDevice,
    fields: FieldBuffers & { bg_thermo: GPUBuffer },
    dims: SimDims,
    params: Params,
    computeRhs: ComputeRhsArtifacts, // <<— new: take the buildRHS artifact
}): RK2Ctx {
    const { device, fields, dims, computeRhs } = opts;
    const N = dims.nx * dims.ny * dims.nz;
    const wg = Math.ceil(N / 256);
    const bytes = N * 4;

    // helper to make zero-inited STORAGE buffers
    const makeBuf = (bytes: number) => {
        const b = device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        });
        new Uint8Array(b.getMappedRange()).fill(0);
        b.unmap();
        return b;
    };

    // star (provisional) state
    const u_star = makeBuf(bytes);
    const v_star = makeBuf(bytes);
    const w_star = makeBuf(bytes);
    const theta_p_star = makeBuf(bytes);
    const qv_star = makeBuf(bytes);
    const qc_star = makeBuf(bytes);

    // archive rhs1
    const rhs1_u = makeBuf(bytes);
    const rhs1_v = makeBuf(bytes);
    const rhs1_w = makeBuf(bytes);
    const rhs1_theta_p = makeBuf(bytes);
    const rhs1_qv = makeBuf(bytes);
    const rhs1_qc = makeBuf(bytes);

    // finals to avoid aliasing
    const w_new = makeBuf(bytes);
    const u_new = makeBuf(bytes);
    const v_new = makeBuf(bytes);
    const theta_p_new = makeBuf(bytes);
    const qv_new = makeBuf(bytes);
    const qc_new = makeBuf(bytes);

    // uniforms
    const U_N = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const U_ax = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    {
        const ab = new ArrayBuffer(32);
        new Uint32Array(ab)[0] = N >>> 0;
        device.queue.writeBuffer(U_N, 0, ab);
    }
    const writeAlpha = (a: number) => {
        const ab = new ArrayBuffer(32);
        const u32 = new Uint32Array(ab);
        const f32 = new Float32Array(ab);
        u32[0] = N >>> 0;
        f32[1] = a;
        device.queue.writeBuffer(U_ax, 0, ab);
    };

    const micro = makeMicrophysicsSaturationAdjust({ device, dims });

    // utility pipelines
    const pipeClear = device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: CLEAR_WGSL }), entryPoint: "main" }
    });
    const pipeCopy = device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: COPY_WGSL }), entryPoint: "main" }
    });
    const pipeAxpy = device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: AXPY_WGSL }), entryPoint: "main" }
    });

    // bindgroups that never change
    const bgClear_rhs_u = device.createBindGroup({
        layout: pipeClear.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: fields.rhs_u } }, { binding: 1, resource: { buffer: U_N } }]
    });
    const bgClear_rhs_v = device.createBindGroup({
        layout: pipeClear.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: fields.rhs_v } }, { binding: 1, resource: { buffer: U_N } }]
    });

    const bgClear_rhs_w = device.createBindGroup({
        layout: pipeClear.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.rhs_w } },
            { binding: 1, resource: { buffer: U_N } },
        ]
    });
    const bgClear_rhs_theta = device.createBindGroup({
        layout: pipeClear.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.rhs_theta_p } },
            { binding: 1, resource: { buffer: U_N } },
        ]
    });
    const bgClear_rhs_qv = device.createBindGroup({
        layout: pipeClear.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.rhs_qv } },
            { binding: 1, resource: { buffer: U_N } },
        ],
    });
    const bgClear_rhs_qc = device.createBindGroup({
        layout: pipeClear.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.rhs_qc } },
            { binding: 1, resource: { buffer: U_N } },
        ],
    });
    // copy rhs → rhs1
    const bgCopy_rhs_u_to_rhs1 = device.createBindGroup({
        layout: pipeCopy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.rhs_u } },
            { binding: 1, resource: { buffer: rhs1_u } },
            { binding: 2, resource: { buffer: U_N } },
        ],
    });

    const bgCopy_rhs_v_to_rhs1 = device.createBindGroup({
        layout: pipeCopy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.rhs_v } },
            { binding: 1, resource: { buffer: rhs1_v } },
            { binding: 2, resource: { buffer: U_N } },
        ],
    });
    const bgCopy_rhs_w_to_rhs1 = device.createBindGroup({
        layout: pipeCopy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.rhs_w } },
            { binding: 1, resource: { buffer: rhs1_w } },
            { binding: 2, resource: { buffer: U_N } },
        ]
    });
    const bgCopy_rhs_th_to_rhs1 = device.createBindGroup({
        layout: pipeCopy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.rhs_theta_p } },
            { binding: 1, resource: { buffer: rhs1_theta_p } },
            { binding: 2, resource: { buffer: U_N } },
        ]
    });

    const bgCopy_rhs_qv_to_rhs1 = device.createBindGroup({
        layout: pipeCopy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.rhs_qv } },
            { binding: 1, resource: { buffer: rhs1_qv } },
            { binding: 2, resource: { buffer: U_N } },
        ],
    });

    const bgCopy_rhs_qc_to_rhs1 = device.createBindGroup({
        layout: pipeCopy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.rhs_qc } },
            { binding: 1, resource: { buffer: rhs1_qc } },
            { binding: 2, resource: { buffer: U_N } },
        ],
    });


    // copy qv/qc -> qv_star/qc_star
    const bgCopy_qv_to_star = device.createBindGroup({
        layout: pipeCopy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.qv } },
            { binding: 1, resource: { buffer: qv_star } },
            { binding: 2, resource: { buffer: U_N } },
        ]
    });
    const bgCopy_qc_to_star = device.createBindGroup({
        layout: pipeCopy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.qc } },
            { binding: 1, resource: { buffer: qc_star } },
            { binding: 2, resource: { buffer: U_N } },
        ]
    });

    // s★ = s0 + dt * rhs1
    const bgAxpy_u_star_from_s0_rhs1 = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.u } },
            { binding: 1, resource: { buffer: rhs1_u } },
            { binding: 2, resource: { buffer: u_star } },
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });

    const bgAxpy_v_star_from_s0_rhs1 = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.v } },
            { binding: 1, resource: { buffer: rhs1_v } },
            { binding: 2, resource: { buffer: v_star } },
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });

    const bgAxpy_w_star_from_s0_rhs1 = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.w } },
            { binding: 1, resource: { buffer: rhs1_w } },
            { binding: 2, resource: { buffer: w_star } },
            { binding: 3, resource: { buffer: U_ax } },
        ]
    });
    const bgAxpy_th_star_from_s0_rhs1 = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.theta_p } },
            { binding: 1, resource: { buffer: rhs1_theta_p } },
            { binding: 2, resource: { buffer: theta_p_star } },
            { binding: 3, resource: { buffer: U_ax } },
        ]
    });
    const bgAxpy_qv_star_from_s0_rhs1 = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.qv } },  // x = qv0
            { binding: 1, resource: { buffer: rhs1_qv } },    // y = rhs1_qv
            { binding: 2, resource: { buffer: qv_star } },    // out = qv_star
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });

    const bgAxpy_qc_star_from_s0_rhs1 = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.qc } },
            { binding: 1, resource: { buffer: rhs1_qc } },
            { binding: 2, resource: { buffer: qc_star } },
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });


    // tmp = rhs1 + rhs2  (use star buffers as tmp)
    // tmp = rhs1 + rhs2  (use star buffers as tmp) — extend to u and v
    const bgAxpy_tmp_u_sum = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: rhs1_u } },
            { binding: 1, resource: { buffer: fields.rhs_u } },  // rhs2_u
            { binding: 2, resource: { buffer: u_star } },        // out: tmp_u
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });

    const bgAxpy_tmp_v_sum = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: rhs1_v } },
            { binding: 1, resource: { buffer: fields.rhs_v } },  // rhs2_v
            { binding: 2, resource: { buffer: v_star } },        // out: tmp_v
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });
    const bgAxpy_tmp_w_sum = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: rhs1_w } },
            { binding: 1, resource: { buffer: fields.rhs_w } },
            { binding: 2, resource: { buffer: w_star } },
            { binding: 3, resource: { buffer: U_ax } },
        ]
    });
    const bgAxpy_tmp_th_sum = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: rhs1_theta_p } },
            { binding: 1, resource: { buffer: fields.rhs_theta_p } },
            { binding: 2, resource: { buffer: theta_p_star } },
            { binding: 3, resource: { buffer: U_ax } },
        ]
    });

    const bgAxpy_tmp_qv_sum = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: rhs1_qv } },
            { binding: 1, resource: { buffer: fields.rhs_qv } },  // rhs2_qv
            { binding: 2, resource: { buffer: qv_star } },        // out: tmp_qv
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });

    const bgAxpy_tmp_qc_sum = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: rhs1_qc } },
            { binding: 1, resource: { buffer: fields.rhs_qc } },  // rhs2_qc
            { binding: 2, resource: { buffer: qc_star } },        // out: tmp_qc
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });


    // FINAL: s_new = s0 + c * tmp  (write new, then blit back)
    const bgAxpy_w_final = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.w } },   // x = s0
            { binding: 1, resource: { buffer: w_star } },     // y = tmp
            { binding: 2, resource: { buffer: w_new } },      // out
            { binding: 3, resource: { buffer: U_ax } },
        ]
    });
    const bgAxpy_th_final = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.theta_p } },
            { binding: 1, resource: { buffer: theta_p_star } },
            { binding: 2, resource: { buffer: theta_p_new } },
            { binding: 3, resource: { buffer: U_ax } },
        ]
    });
    const bgAxpy_u_final = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.u } },   // s0
            { binding: 1, resource: { buffer: u_star } },     // tmp sum
            { binding: 2, resource: { buffer: u_new } },      // out
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });

    const bgAxpy_v_final = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.v } },
            { binding: 1, resource: { buffer: v_star } },
            { binding: 2, resource: { buffer: v_new } },
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });

    const bgAxpy_qv_final = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.qv } },  // x = qv0
            { binding: 1, resource: { buffer: qv_star } },    // y = tmp_qv = rhs1+rhs2
            { binding: 2, resource: { buffer: qv_new } },     // out
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });

    const bgAxpy_qc_final = device.createBindGroup({
        layout: pipeAxpy.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: fields.qc } },
            { binding: 1, resource: { buffer: qc_star } },
            { binding: 2, resource: { buffer: qc_new } },
            { binding: 3, resource: { buffer: U_ax } },
        ],
    });



    // --- helpers -------------------------------------------------------------

    // Clear all RHS accumulators we currently use in buildRHS
    function clearAllRhs(pass: GPUComputePassEncoder) {
        pass.setPipeline(pipeClear); pass.setBindGroup(0, bgClear_rhs_u); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeClear); pass.setBindGroup(0, bgClear_rhs_v); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeClear); pass.setBindGroup(0, bgClear_rhs_w); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeClear); pass.setBindGroup(0, bgClear_rhs_theta); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeClear); pass.setBindGroup(0, bgClear_rhs_qv); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeClear); pass.setBindGroup(0, bgClear_rhs_qc); pass.dispatchWorkgroups(wg);
    }

    // Build RHS at the base state s0 (explicit, even though defaults would work)
    function buildRHS_s0(pass: GPUComputePassEncoder) {
        computeRhs.buildRHS(pass, {
            state: {
                u: fields.u, v: fields.v, w: fields.w,
                theta_p: fields.theta_p, qv: fields.qv, qc: fields.qc,
            },
            out: {
                rhs_u: fields.rhs_u, rhs_v: fields.rhs_v, rhs_w: fields.rhs_w,
                rhs_theta_p: fields.rhs_theta_p, rhs_qv: fields.rhs_qv, rhs_qc: fields.rhs_qc,
            },
        });
    }

    // Build RHS at the star state s★ (writes to the same cleared RHS buffers)
    function buildRHS_star(pass: GPUComputePassEncoder) {
        computeRhs.buildRHS(pass, {
            state: {
                u: u_star, v: v_star, w: w_star,
                theta_p: theta_p_star, qv: qv_star, qc: qc_star,
            },
            // out: same fields.rhs_* as s0 (we clear before calling)
        });
    }

    const projection = makeProjection({ device, dims });
    const projection4 = makeProjectionFD4({device, dims})
    function step(encoder: GPUCommandEncoder, dt: number) {
        // --- 1) set alpha = dt BEFORE opening any pass
        writeAlpha(dt); // writes to U_ax (uniform) — SAFE here

        // ----- PASS A: rhs1 + build star -----
        {
            const pass = encoder.beginComputePass();

            // rhs1: clear → buildRHS(s0)
            clearAllRhs(pass);
            buildRHS_s0(pass);

            // snapshot rhs1
            pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_rhs_u_to_rhs1); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_rhs_v_to_rhs1); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_rhs_w_to_rhs1); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_rhs_th_to_rhs1); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_rhs_qv_to_rhs1); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_rhs_qc_to_rhs1); pass.dispatchWorkgroups(wg);

            // s★ = s0 + dt * rhs1
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_u_star_from_s0_rhs1); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_v_star_from_s0_rhs1); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_w_star_from_s0_rhs1); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_th_star_from_s0_rhs1); pass.dispatchWorkgroups(wg);

            // qv★, qc★ = copies of s0
            // pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_qv_to_star); pass.dispatchWorkgroups(wg);
            // pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_qc_to_star); pass.dispatchWorkgroups(wg);

            // qv★, qc★ = s0 + dt * rhs1
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_qv_star_from_s0_rhs1); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_qc_star_from_s0_rhs1); pass.dispatchWorkgroups(wg);

            // Microphysics on star state: (theta_p★, qv★, qc★)
            // micro.microphysics(
            //     pass,
            //     theta_p_star,   // provisional theta'
            //     qv_star,        // provisional qv
            //     qc_star,        // provisional qc
            //     fields.theta0,  // background θ₀
            //     fields.p0       // background p₀
            // );

            // projection.project(pass, u_star, v_star, w_star, 100)
            projection4.project(pass, u_star, v_star, w_star)
            
            pass.end(); // <<< close pass before changing alpha
        }
        
        // --- 2) set alpha = 1.0 BETWEEN passes
        writeAlpha(1.0);

        // ----- PASS B: rhs2 at star + sum rhs -----
        {
            const pass = encoder.beginComputePass();

            // rhs2: clear → buildRHS(s★)
            clearAllRhs(pass);
            buildRHS_star(pass);

            // tmp = rhs1 + rhs2  (uses star buffers as tmp)
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_tmp_u_sum); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_tmp_v_sum); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_tmp_w_sum); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_tmp_th_sum); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_tmp_qv_sum); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_tmp_qc_sum); pass.dispatchWorkgroups(wg);

            pass.end(); // <<< close pass before changing alpha again
        }

        // --- 3) set alpha = 0.5*dt BETWEEN passes
        writeAlpha(0.5 * dt);

        // ----- PASS C: final combine -----
        {
            const pass = encoder.beginComputePass();

            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_u_final); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_v_final); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_w_final); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_th_final); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_qv_final); pass.dispatchWorkgroups(wg);
            pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_qc_final); pass.dispatchWorkgroups(wg);

            // Microphysics on FINAL state: (theta_p_new, qv, qc)
            // - theta_p_new is your final θ' buffer
            // - fields.qv / fields.qc are the prognostic moisture fields
            // micro.microphysics(
            //     pass,
            //     theta_p_new,
            //     qv_new,
            //     qc_new,
            //     fields.theta0,
            //     fields.p0
            // );

            // Now project final velocities
            // projection.project(pass, u_new, v_new, w_new, 100);
            projection4.project(pass, u_new, v_new, w_new);
            
            pass.end();
        }

        // copy finals back to s0
        encoder.copyBufferToBuffer(u_new, 0, fields.u, 0, bytes);
        encoder.copyBufferToBuffer(v_new, 0, fields.v, 0, bytes);
        encoder.copyBufferToBuffer(w_new, 0, fields.w, 0, bytes);
        encoder.copyBufferToBuffer(theta_p_new, 0, fields.theta_p, 0, bytes);
        encoder.copyBufferToBuffer(qv_new, 0, fields.qv, 0, bytes);
        encoder.copyBufferToBuffer(qc_new, 0, fields.qc, 0, bytes);

    }



    return { step };
}
