// step_rk2.ts
import type { SimDims, FieldBuffers, Params } from "./init_boussinesq";
import type { ComputeRhsArtifacts } from "./compute_rhs";

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
    const rhs1_w = makeBuf(bytes);
    const rhs1_theta_p = makeBuf(bytes);

    // finals to avoid aliasing
    const w_new = makeBuf(bytes);
    const theta_p_new = makeBuf(bytes);

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

    // tmp = rhs1 + rhs2  (use star buffers as tmp)
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


    function step(encoder: GPUCommandEncoder, dt: number) {
        const pass = encoder.beginComputePass();

        // 1) BCs(s0) — later

        // 2) rhs1: clear → buildRHS(s0)
        clearAllRhs(pass);
        buildRHS_s0(pass);

        // 3) snapshot rhs1 (keep your existing snapshots as-is)
        pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_rhs_w_to_rhs1); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_rhs_th_to_rhs1); pass.dispatchWorkgroups(wg);
        // (If/when you advance u/v, add snapshots for rhs_u/rhs_v too.)

        // 4) s★ = s0 + dt * rhs1
        writeAlpha(dt);
        pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_w_star_from_s0_rhs1); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_th_star_from_s0_rhs1); pass.dispatchWorkgroups(wg);

        // qv★, qc★ = copies of s0 (until you add qv/qc RHS)
        pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_qv_to_star); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeCopy); pass.setBindGroup(0, bgCopy_qc_to_star); pass.dispatchWorkgroups(wg);

        // 5) microphysics(s★), BCs(s★) — later
        // 6) projection — later

        // 7) rhs2 at s★: clear → buildRHS(s★)
        clearAllRhs(pass);
        buildRHS_star(pass);

        // 8) final combine: s_new = s0 + 0.5*dt * (rhs1 + rhs2)
        writeAlpha(1.0);
        pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_tmp_w_sum); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_tmp_th_sum); pass.dispatchWorkgroups(wg);

        writeAlpha(0.5 * dt);
        pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_w_final); pass.dispatchWorkgroups(wg);
        pass.setPipeline(pipeAxpy); pass.setBindGroup(0, bgAxpy_th_final); pass.dispatchWorkgroups(wg);

        pass.end();

        // 9) copy finals back to s0 (so defaults remain valid)
        encoder.copyBufferToBuffer(w_new, 0, fields.w, 0, bytes);
        encoder.copyBufferToBuffer(theta_p_new, 0, fields.theta_p, 0, bytes);
    }


    return { step };
}
