// step_rk2.ts
import type { SimDims, FieldBuffers, Params } from "./init_boussinesq";

// small WGSL utils
const CLEAR_WGSL = /*wgsl*/`
struct Uniforms {
  N: u32,
  _pad: vec3<u32>,
};

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
// ---- COPY.wgsl ----
struct CopyUniforms {
  N: u32,
  _pad: vec3<u32>,
};

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
// ---- AXPY.wgsl ----
struct AxpyUniforms {
  N: u32,
  alpha: f32,
  _pad: vec2<f32>,
};

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

type RK2Ctx = {
  step: (encoder: GPUCommandEncoder, dt: number) => void;
};

export function makeStepRK2(opts: {
  device: GPUDevice,
  fields: FieldBuffers & { bg_thermo: GPUBuffer }, // packed [theta0|dthdz|qv_bg]
  dims: SimDims,
  params: Params,
  computeRhsPipeline: GPUComputePipeline,
  computeRhsBind_s0: GPUBindGroup,
  rhsUniformBuf: GPUBuffer
}): RK2Ctx {
  const { device, fields, dims, params, computeRhsPipeline, computeRhsBind_s0, rhsUniformBuf } = opts;
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
  const w_star        = makeBuf(bytes);
  const theta_p_star  = makeBuf(bytes);
  const qv_star       = makeBuf(bytes);
  const qc_star       = makeBuf(bytes);

  // archive rhs1
  const rhs1_w        = makeBuf(bytes);
  const rhs1_theta_p  = makeBuf(bytes);

  // NEW: final buffers to avoid in-place aliasing
  const w_new         = makeBuf(bytes);
  const theta_p_new   = makeBuf(bytes);

  // tiny uniforms (32B due to alignment)
  const U_N  = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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

  // FINAL: s_new = s0 + c * tmp  (write into new buffers to avoid aliasing)
  const bgAxpy_w_final = device.createBindGroup({
    layout: pipeAxpy.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: fields.w } },     // x = s0 (read)
      { binding: 1, resource: { buffer: w_star } },       // y = tmp (read)
      { binding: 2, resource: { buffer: w_new } },        // out = w_new (write)
      { binding: 3, resource: { buffer: U_ax } },
    ]
  });
  const bgAxpy_th_final = device.createBindGroup({
    layout: pipeAxpy.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: fields.theta_p } },   // x = s0
      { binding: 1, resource: { buffer: theta_p_star } },     // y = tmp
      { binding: 2, resource: { buffer: theta_p_new } },      // out = theta_p_new
      { binding: 3, resource: { buffer: U_ax } },
    ]
  });

  // computeRHS at s★
  const computeRhsBind_sStar = device.createBindGroup({
    layout: computeRhsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: theta_p_star } },
      { binding: 1, resource: { buffer: qv_star } },
      { binding: 2, resource: { buffer: qc_star } },
      { binding: 3, resource: { buffer: w_star } },
      { binding: 4, resource: { buffer: fields.rhs_w } },
      { binding: 5, resource: { buffer: fields.rhs_theta_p } },
      { binding: 6, resource: { buffer: (fields as any).bg_thermo } },
      { binding: 7, resource: { buffer: rhsUniformBuf } },
    ],
  });

  function step(encoder: GPUCommandEncoder, dt: number) {
    const pass = encoder.beginComputePass();

    // 1) BCs(s0) — add later

    // 2) rhs1: zero → computeRHS(s0)
    pass.setPipeline(pipeClear);  pass.setBindGroup(0, bgClear_rhs_w);     pass.dispatchWorkgroups(wg);
    pass.setPipeline(pipeClear);  pass.setBindGroup(0, bgClear_rhs_theta); pass.dispatchWorkgroups(wg);

    pass.setPipeline(computeRhsPipeline);
    pass.setBindGroup(0, computeRhsBind_s0);
    pass.dispatchWorkgroups(wg);

    // 3) snapshot rhs1
    pass.setPipeline(pipeCopy);   pass.setBindGroup(0, bgCopy_rhs_w_to_rhs1);    pass.dispatchWorkgroups(wg);
    pass.setPipeline(pipeCopy);   pass.setBindGroup(0, bgCopy_rhs_th_to_rhs1);   pass.dispatchWorkgroups(wg);

    // 4) s★ = s0 + dt * rhs1
    writeAlpha(dt);
    pass.setPipeline(pipeAxpy);   pass.setBindGroup(0, bgAxpy_w_star_from_s0_rhs1);  pass.dispatchWorkgroups(wg);
    pass.setPipeline(pipeAxpy);   pass.setBindGroup(0, bgAxpy_th_star_from_s0_rhs1); pass.dispatchWorkgroups(wg);

    // qv★, qc★ = copies of s0 (until you add qv/qc RHS)
    pass.setPipeline(pipeCopy);   pass.setBindGroup(0, bgCopy_qv_to_star); pass.dispatchWorkgroups(wg);
    pass.setPipeline(pipeCopy);   pass.setBindGroup(0, bgCopy_qc_to_star); pass.dispatchWorkgroups(wg);

    // 5) microphysics(s★), BCs(s★) — add later
    // 6) projection — add later

    // 7) rhs2 at s★: zero → computeRHS(s★)
    pass.setPipeline(pipeClear);  pass.setBindGroup(0, bgClear_rhs_w);     pass.dispatchWorkgroups(wg);
    pass.setPipeline(pipeClear);  pass.setBindGroup(0, bgClear_rhs_theta); pass.dispatchWorkgroups(wg);

    pass.setPipeline(computeRhsPipeline);
    pass.setBindGroup(0, computeRhsBind_sStar);
    pass.dispatchWorkgroups(wg);

    // 8) final combine: s_new = s0 + 0.5*dt * (rhs1 + rhs2)
    writeAlpha(1.0);
    pass.setPipeline(pipeAxpy);   pass.setBindGroup(0, bgAxpy_tmp_w_sum);  pass.dispatchWorkgroups(wg);
    pass.setPipeline(pipeAxpy);   pass.setBindGroup(0, bgAxpy_tmp_th_sum); pass.dispatchWorkgroups(wg);

    writeAlpha(0.5 * dt);
    pass.setPipeline(pipeAxpy);   pass.setBindGroup(0, bgAxpy_w_final);    pass.dispatchWorkgroups(wg);
    pass.setPipeline(pipeAxpy);   pass.setBindGroup(0, bgAxpy_th_final);   pass.dispatchWorkgroups(wg);

    // 9) microphysics(sₙ), BCs(sₙ) — add later
    pass.end();

    // copy finals back to s0 buffers (keeps your prebuilt bindgroups valid)
    encoder.copyBufferToBuffer(w_new, 0, fields.w, 0, bytes);
    encoder.copyBufferToBuffer(theta_p_new, 0, fields.theta_p, 0, bytes);
  }

  return { step };
}
