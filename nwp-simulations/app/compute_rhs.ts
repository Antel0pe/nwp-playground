// compute_rhs.ts
import type { SimDims, FieldBuffers, Params } from "./init_boussinesq";

export type ComputeRhsArtifacts = {
  pipeline: GPUComputePipeline;
  bind_s0: GPUBindGroup;     // bindgroup pointing at the s0 state
  rhsUniformBuf: GPUBuffer;  // uniforms buffer (shared so RK2 can build s★ bindgroups)
};

export function makeComputeRhs(opts: {
  device: GPUDevice;
  fields: FieldBuffers & { bg_thermo: GPUBuffer }; // packed [theta0|dthdz|qv_bg]
  dims: SimDims;
  params: Params;
}): ComputeRhsArtifacts {
  const { device, fields, dims, params } = opts;
  const { nx, ny, nz } = dims;

  const computeRhsModule = device.createShaderModule({
    code: /* wgsl */ `
// --- computeRHS.wgsl ---

struct Uniforms {
  N: u32,         // total cells
  offTheta0: u32, // offset (in elements) for theta0
  offDthdz: u32,  // offset for dtheta0_dz
  offQvbg: u32,   // offset for qv_bg
  g: f32,         // gravity
  _pad: vec3<f32>
};

@group(0) @binding(0) var<storage, read>        theta_p      : array<f32>;
@group(0) @binding(1) var<storage, read>        qv           : array<f32>;
@group(0) @binding(2) var<storage, read>        qc           : array<f32>;
@group(0) @binding(3) var<storage, read>        w_in         : array<f32>;
@group(0) @binding(4) var<storage, read_write>  rhs_w        : array<f32>;
@group(0) @binding(5) var<storage, read_write>  rhs_theta_p  : array<f32>;

// Packed background fields: [ theta0 | dtheta0_dz | qv_bg ]
@group(0) @binding(6) var<storage, read>        bg           : array<f32>;
@group(0) @binding(7) var<uniform>              uniforms     : Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uniforms.N) { return; }

  let thp   = theta_p[i];
  let qv_i  = qv[i];
  let qc_i  = qc[i];
  let w_i   = w_in[i];

  let th0   = bg[uniforms.offTheta0 + i];
  let dthdz = bg[uniforms.offDthdz + i];
  let qvbg  = bg[uniforms.offQvbg  + i];

  // b = g * ( thp/th0 + 0.61*(qv - qv_bg) - qc )
  let b = uniforms.g * ( (thp / th0) + 0.61 * (qv_i - qvbg) - qc_i );

  rhs_w[i]       = rhs_w[i] + b;
  rhs_theta_p[i] = rhs_theta_p[i] + (-w_i * dthdz);
}
`,
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: computeRhsModule, entryPoint: "main" },
  });

  // ---- uniforms
  const N = nx * ny * nz;
  const offTheta0 = 0;
  const offDthdz = N;
  const offQvbg = 2 * N;

  // 48 bytes here is fine (>= 32B; std140-like alignment); we only use the first 32B in WGSL
  const rhsUniformsAB = new ArrayBuffer(48);
  const u32 = new Uint32Array(rhsUniformsAB);
  const f32 = new Float32Array(rhsUniformsAB);
  u32[0] = N >>> 0;
  u32[1] = offTheta0 >>> 0;
  u32[2] = offDthdz >>> 0;
  u32[3] = offQvbg >>> 0;
  f32[4] = (params.g ?? 9.81);

  const rhsUniformBuf = device.createBuffer({
    size: rhsUniformsAB.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(rhsUniformBuf, 0, rhsUniformsAB);

  // ---- bindgroup for the initial state s0
  const bind_s0 = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: fields.theta_p } },
      { binding: 1, resource: { buffer: fields.qv } },
      { binding: 2, resource: { buffer: fields.qc } },
      { binding: 3, resource: { buffer: fields.w } },
      { binding: 4, resource: { buffer: fields.rhs_w } },
      { binding: 5, resource: { buffer: fields.rhs_theta_p } },
      { binding: 6, resource: { buffer: fields.bg_thermo } },
      { binding: 7, resource: { buffer: rhsUniformBuf } },
    ],
  });

  return { pipeline, bind_s0, rhsUniformBuf };
}
