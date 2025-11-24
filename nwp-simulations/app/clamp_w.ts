// clamp_w.ts
import type { SimDims } from "./init_boussinesq";

const WG = 256;

export type ClampWIO = {
  w: GPUBuffer; // vertical velocity field, clamped in-place at top/bottom
};

export type ClampWArtifacts = {
  dispatch: (pass: GPUComputePassEncoder, w: GPUBuffer) => void;
  resources: {
    pipeline: GPUComputePipeline;
    uniforms: GPUBuffer;
    workgroups: number;
  };
};

/**
 * Zero out w at rigid lid/floor (iz == 0 or iz == nz-1).
 * Call this after advection/forces and after projection if you want
 * to strictly enforce w=0 at boundaries each step.
 */
export function makeClampW(opts: {
  device: GPUDevice;
  dims: SimDims;
}): ClampWArtifacts {
  const { device, dims } = opts;
  const { nx, ny, nz, dx, dy, dz } = dims;
  const N = nx * ny * nz;

  const module = device.createShaderModule({
    label: "clamp_w_module",
    code: /* wgsl */`
struct FD4U {
  nx: u32,
  ny: u32,
  nz: u32,
  N:  u32,
  sx: u32,
  sy: u32,
  sz: u32,
  _pad: u32,
  invdx12: f32,
  invdy12: f32,
  invdz12: f32,
  _padf: f32,
};

@group(0) @binding(0) var<storage, read_write> w  : array<f32>;
@group(0) @binding(1) var<uniform> FD4 : FD4U;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= FD4.N) { return; }

  let nx = FD4.nx;
  let ny = FD4.ny;
  let nz = FD4.nz;

  let iz = i / (nx * ny);

  if (iz == 0u || iz == nz - 1u) {
    w[i] = 0.0;
  }
}
`,
  });

  const pipeline = device.createComputePipeline({
    label: "clamp_w_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  // FD4U uniforms: 12 * 4 bytes = 48 bytes
  // We keep the same layout you use elsewhere (nx,ny,nz,N,sx,sy,sz,..., invdx12, invdy12, invdz12)
  const ab = new ArrayBuffer(48);
  const u32 = new Uint32Array(ab);
  const f32 = new Float32Array(ab);

  u32[0] = nx >>> 0;
  u32[1] = ny >>> 0;
  u32[2] = nz >>> 0;
  u32[3] = N  >>> 0;
  u32[4] = 1;              // sx
  u32[5] = nx >>> 0;       // sy
  u32[6] = (nx * ny) >>> 0; // sz
  u32[7] = 0;              // _pad

  // invdx12 etc aren't used by this kernel, but we fill them consistently.
  f32[8]  = 1.0 / (12.0 * dx);
  f32[9]  = 1.0 / (12.0 * dy);
  f32[10] = 1.0 / (12.0 * dz);
  f32[11] = 0.0; // _padf

  const uniforms = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "clamp_w_uniforms",
  });
  device.queue.writeBuffer(uniforms, 0, ab);

  const workgroups = Math.ceil(N / WG);

  function dispatch(pass: GPUComputePassEncoder, w: GPUBuffer) {
    const bg = device.createBindGroup({
      label: "clamp_w_bg",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: w } },
        { binding: 1, resource: { buffer: uniforms } },
      ],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups);
  }

  return { dispatch, resources: { pipeline, uniforms, workgroups } };
}
