// jacobi_poisson.ts
import type { SimDims } from "../init_boussinesq";

export type JacobiIO = {
  rhs_div: GPUBuffer;     // b = div
  psi_in: GPUBuffer;      // ψ^n
  psi_out: GPUBuffer;     // ψ^{n+1}
};

export function makeJacobiPoisson(opts:{ device: GPUDevice; dims: SimDims }) {
  const { device, dims } = opts;
  const { nx, ny, nz, dx, dy, dz } = dims;
  const N = nx*ny*nz;

  const ax = 1.0/(dx*dx), ay = 1.0/(dy*dy), az = 1.0/(dz*dz);
  const denom = 2.0*(ax+ay+az);

  const module = device.createShaderModule({
    label: "jacobi_poisson_module",
    code: /* wgsl */`
struct U {
  nx: u32,
  ny: u32,
  nz: u32,
  N:  u32,
  sx: u32,
  sy: u32,
  sz: u32,
  _pad: u32,
  ax: f32,
  ay: f32,
  az: f32,
  invDen: f32,
};

@group(0) @binding(0) var<storage, read>       b     : array<f32>;
@group(0) @binding(1) var<storage, read>       psi_n : array<f32>;
@group(0) @binding(2) var<storage, read_write> psi_np: array<f32>;
@group(0) @binding(3) var<uniform>             Ubuf  : U;

fn wrap_inc(i:u32,n:u32)->u32{ return select(i+1u,0u,i+1u>=n); }
fn wrap_dec(i:u32,n:u32)->u32{ return select(i-1u,n-1u,i==0u); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let id = gid.x; if (id>=Ubuf.N){return;}
  let nx=Ubuf.nx; let ny=Ubuf.ny;
  let sx=Ubuf.sx; let sy=Ubuf.sy; let sz=Ubuf.sz;

  let ix = id % nx;
  let iy = (id / nx) % ny;
  let iz = id / (nx*ny);

  let ixp=wrap_inc(ix,nx); let ixm=wrap_dec(ix,nx);
  let iyp=wrap_inc(iy,ny); let iym=wrap_dec(iy,ny);
  let izp=wrap_inc(iz,Ubuf.nz); let izm=wrap_dec(iz,Ubuf.nz);

  let i_xp = iz*sz + iy*sy + ixp*sx;
  let i_xm = iz*sz + iy*sy + ixm*sx;
  let i_yp = iz*sz + iyp*sy + ix*sx;
  let i_ym = iz*sz + iym*sy + ix*sx;
  let i_zp = izp*sz + iy*sy + ix*sx;
  let i_zm = izm*sz + iy*sy + ix*sx;

  let sumN =
      Ubuf.ax*(psi_n[i_xp]+psi_n[i_xm]) +
      Ubuf.ay*(psi_n[i_yp]+psi_n[i_ym]) +
      Ubuf.az*(psi_n[i_zp]+psi_n[i_zm]);

  // Jacobi: ψ^{n+1} = (sumN - b) / (2(ax+ay+az))
  psi_np[id] = (sumN - b[id]) * Ubuf.invDen;
}
`
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label: "jacobi_poisson_pipeline",
  });

  const ab = new ArrayBuffer(48);
  const u32 = new Uint32Array(ab);
  const f32 = new Float32Array(ab);
  u32[0]=nx; u32[1]=ny; u32[2]=nz; u32[3]=N;
  u32[4]=1;  u32[5]=nx; u32[6]=nx*ny; u32[7]=0;
  f32[8]=ax; f32[9]=ay; f32[10]=az; f32[11]=1.0/denom;

  const uniforms = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniforms, 0, ab);

  const workgroups = Math.ceil(N/256);

  function dispatch(pass: GPUComputePassEncoder, io: JacobiIO){
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding:0, resource:{ buffer: io.rhs_div } },
        { binding:1, resource:{ buffer: io.psi_in } },
        { binding:2, resource:{ buffer: io.psi_out } },
        { binding:3, resource:{ buffer: uniforms } },
      ],
      label: "jacobi_poisson_bg"
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0,bg);
    pass.dispatchWorkgroups(workgroups);
  }

  return { dispatch, resources: { pipeline, uniforms, workgroups } };
}
