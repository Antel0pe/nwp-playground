// grad_subtract.ts
import type { SimDims } from "../init_boussinesq";

export type GradSubtractIO = {
  psi: GPUBuffer;
  u: GPUBuffer;
  v: GPUBuffer;
  w: GPUBuffer; // corrected in-place
};

export function makeGradSubtract(opts:{ device: GPUDevice; dims: SimDims }) {
  const { device, dims } = opts;
  const { nx, ny, nz, dx, dy, dz } = dims;
  const N = nx * ny * nz;

  const module = device.createShaderModule({
    label: "grad_subtract_module",
    code: /* wgsl */ `
struct U {
  nx: u32,
  ny: u32,
  nz: u32,
  N:  u32,
  sx: u32,
  sy: u32,
  sz: u32,
  _pad: u32,
  invdx: f32,
  invdy: f32,
  invdz: f32,
  _padf: f32,
};

@group(0) @binding(0) var<storage, read>       psi : array<f32>;
@group(0) @binding(1) var<storage, read_write> u   : array<f32>;
@group(0) @binding(2) var<storage, read_write> v   : array<f32>;
@group(0) @binding(3) var<storage, read_write> w   : array<f32>;
@group(0) @binding(4) var<uniform>             Ubuf: U;

fn wrap_inc(i:u32,n:u32)->u32{ return select(i+1u,0u,i+1u>=n); }
fn wrap_dec(i:u32,n:u32)->u32{ return select(i-1u,n-1u,i==0u); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let id = gid.x;
  if (id >= Ubuf.N) { return; }

  let nx = Ubuf.nx;
  let ny = Ubuf.ny;
  let sx = Ubuf.sx;
  let sy = Ubuf.sy;
  let sz = Ubuf.sz;

  let ix = id % nx;
  let iy = (id / nx) % ny;
  let iz = id / (nx * ny);

  let ixp = wrap_inc(ix, nx);
  let iyp = wrap_inc(iy, ny);
  let izp = wrap_inc(iz, Ubuf.nz);

  let i_xp = iz * sz + iy * sy + ixp * sx;
  let i_yp = iz * sz + iyp * sy + ix * sx;
  let i_zp = izp * sz + iy * sy + ix * sx;

  // G_x ψ = (ψ[i+1] - ψ[i]) / dx   (forward)
  let dpsi_dx = (psi[i_xp] - psi[id]) * Ubuf.invdx;
  let dpsi_dy = (psi[i_yp] - psi[id]) * Ubuf.invdy;
  let dpsi_dz = (psi[i_zp] - psi[id]) * Ubuf.invdz;

  u[id] = u[id] - dpsi_dx;
  v[id] = v[id] - dpsi_dy;
  w[id] = w[id] - dpsi_dz;
}
`
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label: "grad_subtract_pipeline",
  });

  // pack uniforms
  const ab = new ArrayBuffer(48);
  const u32 = new Uint32Array(ab);
  const f32 = new Float32Array(ab);

  u32[0] = nx;      // nx
  u32[1] = ny;      // ny
  u32[2] = nz;      // nz
  u32[3] = N;       // N
  u32[4] = 1;       // sx
  u32[5] = nx;      // sy
  u32[6] = nx*ny;   // sz
  u32[7] = 0;       // _pad

  f32[8]  = 1.0 / dx;  // invdx
  f32[9]  = 1.0 / dy;  // invdy
  f32[10] = 1.0 / dz;  // invdz
  f32[11] = 0.0;       // _padf

  const uniforms = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, ab);

  const workgroups = Math.ceil(N / 256);

  function dispatch(pass: GPUComputePassEncoder, io: GradSubtractIO){
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: io.psi } },
        { binding: 1, resource: { buffer: io.u } },
        { binding: 2, resource: { buffer: io.v } },
        { binding: 3, resource: { buffer: io.w } },
        { binding: 4, resource: { buffer: uniforms } },
      ],
      label: "grad_subtract_bg",
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups);
  }

  return { dispatch, resources: { pipeline, uniforms, workgroups } };
}
