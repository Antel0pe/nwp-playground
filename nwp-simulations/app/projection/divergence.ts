// divergence.ts
import type { SimDims } from "../init_boussinesq";

export type DivergenceIO = {
  u: GPUBuffer; v: GPUBuffer; w: GPUBuffer;
  out_div: GPUBuffer;
};

export function makeDivergence(opts: { device: GPUDevice; dims: SimDims }) {
  const { device, dims } = opts;
  const { nx, ny, nz, dx, dy, dz } = dims;
  const N = nx * ny * nz;

  const module = device.createShaderModule({
    label: "divergence_module",
    code: /* wgsl */`
struct U {
  nx: u32, ny: u32, nz: u32, N: u32,
  sx: u32, sy: u32, sz: u32, _pad: u32,
  inv2dx: f32, inv2dy: f32, inv2dz: f32, _padf: f32,
};
@group(0) @binding(0) var<storage, read>  u_in : array<f32>;
@group(0) @binding(1) var<storage, read>  v_in : array<f32>;
@group(0) @binding(2) var<storage, read>  w_in : array<f32>;
@group(0) @binding(3) var<storage, read_write> div : array<f32>;
@group(0) @binding(4) var<uniform> Ubuf : U;

fn wrap_inc(i:u32,n:u32)->u32{ return select(i+1u,0u,i+1u>=n); }
fn wrap_dec(i:u32,n:u32)->u32{ return select(i-1u,n-1u,i==0u); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= Ubuf.N) { return; }
  let nx = Ubuf.nx; let ny = Ubuf.ny;
  let sx = Ubuf.sx; let sy = Ubuf.sy; let sz = Ubuf.sz;

  let ix = id % nx;
  let iy = (id / nx) % ny;
  let iz = id / (nx*ny);

  let ixp = wrap_inc(ix,nx); let ixm = wrap_dec(ix,nx);
  let iyp = wrap_inc(iy,ny); let iym = wrap_dec(iy,ny);
  let izp = wrap_inc(iz,Ubuf.nz); let izm = wrap_dec(iz,Ubuf.nz);

  let i_xp = iz*sz + iy*sy + ixp*sx;
  let i_xm = iz*sz + iy*sy + ixm*sx;
  let i_yp = iz*sz + iyp*sy + ix*sx;
  let i_ym = iz*sz + iym*sy + ix*sx;
  let i_zp = izp*sz + iy*sy + ix*sx;
  let i_zm = izm*sz + iy*sy + ix*sx;

  let du_dx = (u_in[i_xp] - u_in[i_xm]) * Ubuf.inv2dx;
  let dv_dy = (v_in[i_yp] - v_in[i_ym]) * Ubuf.inv2dy;
  let dw_dz = (w_in[i_zp] - w_in[i_zm]) * Ubuf.inv2dz;

  div[id] = du_dx + dv_dy + dw_dz;
}
`
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label: "divergence_pipeline",
  });

  const ab = new ArrayBuffer(48);
  const u32 = new Uint32Array(ab);
  const f32 = new Float32Array(ab);
  u32[0]=nx; u32[1]=ny; u32[2]=nz; u32[3]=N;
  u32[4]=1;  u32[5]=nx; u32[6]=nx*ny; u32[7]=0;
  f32[8] = 1.0/(2.0*dx); f32[9] = 1.0/(2.0*dy); f32[10]=1.0/(2.0*dz); f32[11]=0.0;
  const uniforms = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniforms, 0, ab);

  const workgroups = Math.ceil(N/256);

  function dispatch(pass: GPUComputePassEncoder, io: DivergenceIO){
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {binding:0, resource:{buffer: io.u}},
        {binding:1, resource:{buffer: io.v}},
        {binding:2, resource:{buffer: io.w}},
        {binding:3, resource:{buffer: io.out_div}},
        {binding:4, resource:{buffer: uniforms}},
      ],
      label: "divergence_bg"
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0,bg);
    pass.dispatchWorkgroups(workgroups);
  }

  return { dispatch, resources: { pipeline, uniforms, workgroups } };
}
