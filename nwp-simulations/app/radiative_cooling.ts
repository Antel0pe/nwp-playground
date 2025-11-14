// radiative_cooling.ts
import type { SimDims, Params } from "./init_boussinesq";

export type RadiativeCoolingIO = {
  theta_p: GPUBuffer;
  rhs_theta_p: GPUBuffer;
};

export type RadiativeCoolingArtifacts = {
  dispatch: (pass: GPUComputePassEncoder, io: RadiativeCoolingIO) => void;
  resources: { pipeline: GPUComputePipeline; uniforms: GPUBuffer; workgroups: number };
};

export function makeRadiativeCooling(opts: {
  device: GPUDevice;
  dims: SimDims;
  params: Params;
}): RadiativeCoolingArtifacts {
  const { device, dims, params } = opts;
  const { nx, ny, nz } = dims;
  const N = nx * ny * nz;
  const tau = params.tau_rad ?? 1800.0;

  const wgsl = /* wgsl */`
struct U {
  N: u32,
  inv_tau: f32,
  _pad: vec2<f32>,
};

@group(0) @binding(0) var<storage, read>       theta_p : array<f32>;
@group(0) @binding(1) var<storage, read_write> rhs_th  : array<f32>;
@group(0) @binding(2) var<uniform>             Ubuf    : U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= Ubuf.N) { return; }
  rhs_th[i] = rhs_th[i] - theta_p[i] * Ubuf.inv_tau;
}
`;

  const module = device.createShaderModule({ code: wgsl, label: "radiative_cooling_module" });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label: "radiative_cooling_pipeline",
  });

  const ab = new ArrayBuffer(16);
  const u32 = new Uint32Array(ab);
  const f32 = new Float32Array(ab);
  u32[0] = N >>> 0;
  f32[1] = 1.0 / tau;
  const uniforms = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniforms, 0, ab);

  const workgroups = Math.ceil(N / 256);

  function dispatch(pass: GPUComputePassEncoder, io: RadiativeCoolingIO) {
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: io.theta_p } },
        { binding: 1, resource: { buffer: io.rhs_theta_p } },
        { binding: 2, resource: { buffer: uniforms } },
      ],
      label: "radiative_cooling_bg",
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups);
  }

  return { dispatch, resources: { pipeline, uniforms, workgroups } };
}
