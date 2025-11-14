// surface_relax.ts
import type { SimDims, Params } from "./init_boussinesq";

export type SurfaceRelaxIO = {
  theta0: GPUBuffer;           // background θ0
  theta_p: GPUBuffer;          // current θ′
  qv: GPUBuffer;               // current qv
  theta_target: GPUBuffer;     // theta_surf_target (3D)
  qv_target: GPUBuffer;        // qv_surf_target (3D)
  rhs_theta_p: GPUBuffer;      // accumulate here
  rhs_qv: GPUBuffer;           // accumulate here
};

export type SurfaceRelaxArtifacts = {
  dispatch: (pass: GPUComputePassEncoder, io: SurfaceRelaxIO) => void;
  resources: { pipeline: GPUComputePipeline; uniforms: GPUBuffer; workgroups: number };
};

export function makeSurfaceRelax(opts: {
  device: GPUDevice;
  dims: SimDims;
  params: Params;               // uses Nbl, tau_surf
}): SurfaceRelaxArtifacts {
  const { device, dims, params } = opts;
  const { nx, ny, nz, dx, dy, dz } = dims;
  const N = nx * ny * nz;

  const Nbl = Math.max(0, Math.min(params.Nbl ?? 0, nz)); // clamp to nz
  const inv_tau = 1.0 / (params.tau_surf ?? 100.0);       // always apply as requested

  const module = device.createShaderModule({
    label: "surface_relax_module",
    code: /* wgsl */`
struct U {
  nx: u32, ny: u32, nz: u32, N: u32,
  sx: u32, sy: u32, sz: u32, kmax: u32,
  inv_tau: f32, _pad: vec3<f32>,
};

@group(0) @binding(0) var<storage, read>       theta0       : array<f32>;
@group(0) @binding(1) var<storage, read>       theta_p      : array<f32>;
@group(0) @binding(2) var<storage, read>       qv_in        : array<f32>;
@group(0) @binding(3) var<storage, read>       theta_target : array<f32>;
@group(0) @binding(4) var<storage, read>       qv_target    : array<f32>;
@group(0) @binding(5) var<storage, read_write> rhs_theta_p  : array<f32>;
@group(0) @binding(6) var<storage, read_write> rhs_qv       : array<f32>;
@group(0) @binding(7) var<uniform>             Ubuf         : U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= Ubuf.N) { return; }

  let nx = Ubuf.nx; let ny = Ubuf.ny;
  let ix = i % nx;
  let iy = (i / nx) % ny;
  let iz = i / (nx * ny);

  if (iz >= Ubuf.kmax) { return; } // only bottom kmax levels

  // actual θ in BL: θ = θ0 + θ′
  let theta_bl = theta0[i] + theta_p[i];

  // relax toward targets
  let dtheta = (theta_target[i] - theta_bl) * Ubuf.inv_tau;
  let dqv    = (qv_target[i]    - qv_in[i]) * Ubuf.inv_tau;

  rhs_theta_p[i] = rhs_theta_p[i] + dtheta;
  rhs_qv[i]      = rhs_qv[i]      + dqv;
}
`,
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label: "surface_relax_pipeline",
  });

  const ab = new ArrayBuffer(64);
  const u32 = new Uint32Array(ab);
  const f32 = new Float32Array(ab);
  u32[0] = nx >>> 0;  u32[1] = ny >>> 0;  u32[2] = nz >>> 0;  u32[3] = N >>> 0;
  u32[4] = 1;         u32[5] = nx >>> 0;  u32[6] = (nx*ny) >>> 0; u32[7] = Nbl >>> 0;
  f32[8]  = inv_tau;  // inv_tau in slot 8; rest padding

  const uniforms = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "surface_relax_uniforms" });
  device.queue.writeBuffer(uniforms, 0, ab);

  const workgroups = Math.ceil(N / 256);

  function dispatch(pass: GPUComputePassEncoder, io: SurfaceRelaxIO) {
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: io.theta0 } },
        { binding: 1, resource: { buffer: io.theta_p } },
        { binding: 2, resource: { buffer: io.qv } },
        { binding: 3, resource: { buffer: io.theta_target } },
        { binding: 4, resource: { buffer: io.qv_target } },
        { binding: 5, resource: { buffer: io.rhs_theta_p } },
        { binding: 6, resource: { buffer: io.rhs_qv } },
        { binding: 7, resource: { buffer: uniforms } },
      ],
      label: "surface_relax_bg",
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups);
  }

  return { dispatch, resources: { pipeline, uniforms, workgroups } };
}
