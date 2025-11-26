// project_velocity_fd4.ts
import type { SimDims } from "../init_boussinesq";

const WG = 256;

type Fd4Buffers = {
  bDiv: GPUBuffer;    // divergence RHS
  psi: GPUBuffer;     // solution
  r: GPUBuffer;       // residual
  p: GPUBuffer;       // CG direction
  Ap: GPUBuffer;      // L4(p)
  gx: GPUBuffer;
  gy: GPUBuffer;
  gz: GPUBuffer;

  partials: GPUBuffer; // partial sums for reductions (dot/sum)

  rsold: GPUBuffer;   // 1 float
  rsnew: GPUBuffer;   // 1 float
  pAp: GPUBuffer;     // 1 float
  sumDiv: GPUBuffer;  // 1 float (sum of divergence)
};

type Fd4Uniforms = {
  fd4: GPUBuffer;       // nx,ny,nz,N,sx,sy,sz, invdx12,invdy12,invdz12
  dotU: GPUBuffer;      // N, numWG
  sumU: GPUBuffer;      // N, numWG
  reduceU: GPUBuffer;   // numWG
  meanU: GPUBuffer;     // N
  clearU: GPUBuffer;    // N
  copyU: GPUBuffer;     // N
  projU: GPUBuffer;     // N
};

type Fd4Pipelines = {
  div4: GPUComputePipeline;
  grad4: GPUComputePipeline;
  dot: GPUComputePipeline;
  sum: GPUComputePipeline;
  reduce: GPUComputePipeline;
  subtractMean: GPUComputePipeline;
  clear: GPUComputePipeline;
  copy: GPUComputePipeline;
  updatePsiR: GPUComputePipeline;
  updateP: GPUComputePipeline;
  copyScalar: GPUComputePipeline;
  projectSubtract: GPUComputePipeline;
  mulCoeff: GPUComputePipeline;
  mulCoeffInplace: GPUComputePipeline;
  clampW: GPUComputePipeline;
  grad4_bc: GPUComputePipeline;
};

export function makeProjectionFD4(opts: { device: GPUDevice; dims: SimDims }) {
  const { device, dims } = opts;
  const { nx, ny, nz, dx, dy, dz } = dims;
  const N = nx * ny * nz;
  const bytes = N * 4;
  const numWG = Math.ceil(N / WG);

  // ---------- shared FD4 uniform ----------
  const fd4AB = new ArrayBuffer(48);
  {
    const u32 = new Uint32Array(fd4AB);
    const f32 = new Float32Array(fd4AB);
    u32[0] = nx;
    u32[1] = ny;
    u32[2] = nz;
    u32[3] = N;
    u32[4] = 1;        // sx
    u32[5] = nx;       // sy
    u32[6] = nx * ny;  // sz
    u32[7] = 0;

    const invdx12 = 1.0 / (12.0 * dx);
    const invdy12 = 1.0 / (12.0 * dy);
    const invdz12 = 1.0 / (12.0 * dz);

    f32[8] = invdx12;
    f32[9] = invdy12;
    f32[10] = invdz12;
    f32[11] = 0.0;
  }

  const fd4Uniform = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(fd4Uniform, 0, fd4AB);

  // ---------- buffers ----------
  const buffers: Fd4Buffers = {
    bDiv: device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    psi: device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    r: device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    p: device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    Ap: device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    gx: device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    gy: device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    gz: device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),

    partials: device.createBuffer({
      size: numWG * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),

    rsold: device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    rsnew: device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    pAp: device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    sumDiv: device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
  };
  const readbackPAp = device.createBuffer({
  size: 4,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
const readbackRsold = device.createBuffer({
  size: 4,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
const readbackRsnew = device.createBuffer({
  size: 4,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});


  // ---------- small uniforms ----------
  function makeScalarUniform(size = 16) {
    return device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  const dotUAB = new ArrayBuffer(16);
  {
    const u32 = new Uint32Array(dotUAB);
    u32[0] = N;
    u32[1] = numWG;
  }
  const dotU = makeScalarUniform(16);
  device.queue.writeBuffer(dotU, 0, dotUAB);

  const sumUAB = new ArrayBuffer(16);
  {
    const u32 = new Uint32Array(sumUAB);
    u32[0] = N;
    u32[1] = numWG;
  }
  const sumU = makeScalarUniform(16);
  device.queue.writeBuffer(sumU, 0, sumUAB);

  const reduceUAB = new ArrayBuffer(16);
  {
    const u32 = new Uint32Array(reduceUAB);
    u32[0] = numWG;
  }
  const reduceU = makeScalarUniform(16);
  device.queue.writeBuffer(reduceU, 0, reduceUAB);

  const meanUAB = new ArrayBuffer(16);
  {
    const u32 = new Uint32Array(meanUAB);
    u32[0] = N;
  }
  const meanU = makeScalarUniform(16);
  device.queue.writeBuffer(meanU, 0, meanUAB);

  const clearUAB = new ArrayBuffer(16);
  {
    const u32 = new Uint32Array(clearUAB);
    u32[0] = N;
  }
  const clearU = makeScalarUniform(16);
  device.queue.writeBuffer(clearU, 0, clearUAB);

  const copyUAB = new ArrayBuffer(16);
  {
    const u32 = new Uint32Array(copyUAB);
    u32[0] = N;
  }
  const copyU = makeScalarUniform(16);
  device.queue.writeBuffer(copyU, 0, copyUAB);

  const projUAB = new ArrayBuffer(16);
  {
    const u32 = new Uint32Array(projUAB);
    u32[0] = N;
  }
  const projU = makeScalarUniform(16);
  device.queue.writeBuffer(projU, 0, projUAB);

  const uniforms: Fd4Uniforms = {
    fd4: fd4Uniform,
    dotU,
    sumU,
    reduceU,
    meanU,
    clearU,
    copyU,
    projU,
  };

  // ---------- WGSL common FD4 ----------
  const FD4_COMMON = /* wgsl */`
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

@group(0) @binding(4) var<uniform> FD4 : FD4U;

fn wrap_inc(i:u32,n:u32)->u32{ return select(i+1u,0u,i+1u>=n); }
fn wrap_dec(i:u32,n:u32)->u32{ return select(i-1u,n-1u,i==0u); }

fn idx3(ix:u32, iy:u32, iz:u32)->u32{
  return iz*FD4.sz + iy*FD4.sy + ix*FD4.sx;
}
`;

  const MUL_COEFF_WGSL = /* wgsl */`
struct MulU {
  N: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       coeff : array<f32>;
@group(0) @binding(1) var<storage, read>       src   : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst   : array<f32>;
@group(0) @binding(3) var<uniform>             U     : MulU;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }
  dst[i] = coeff[i] * src[i];
}
`;

  const MUL_COEFF_INPLACE_WGSL = /* wgsl */`
struct MulU {
  N: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       coeff : array<f32>;
@group(0) @binding(1) var<storage, read_write> buf   : array<f32>;
@group(0) @binding(2) var<uniform>             U     : MulU;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }
  buf[i] = coeff[i] * buf[i];
}
`;



  // ---------- div4 ----------
  const DIV4_WGSL = /* wgsl */`
${FD4_COMMON}

@group(0) @binding(0) var<storage, read>       u : array<f32>;
@group(0) @binding(1) var<storage, read>       v : array<f32>;
@group(0) @binding(2) var<storage, read>       w : array<f32>;
@group(0) @binding(3) var<storage, read_write> outDiv : array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= FD4.N) { return; }

  let nx = FD4.nx;
  let ny = FD4.ny;
  let ix = id % nx;
  let iy = (id / nx) % ny;
  let iz = id / (nx * ny);

  let ixp1 = wrap_inc(ix,nx);
  let ixp2 = wrap_inc(ixp1,nx);
  let ixm1 = wrap_dec(ix,nx);
  let ixm2 = wrap_dec(ixm1,nx);

  let iyp1 = wrap_inc(iy,ny);
  let iyp2 = wrap_inc(iyp1,ny);
  let iym1 = wrap_dec(iy,ny);
  let iym2 = wrap_dec(iym1,ny);

  let nz = FD4.nz;
  // let izp1 = wrap_inc(iz,nz);
  // let izp2 = wrap_inc(izp1,nz);
  // let izm1 = wrap_dec(iz,nz);
  // let izm2 = wrap_dec(izm1,nz);
  // let izp1 = min(iz + 1u, nz - 1u);
  // let izp2 = min(iz + 2u, nz - 1u);
  // let izm1 = max(iz - 1u, 0u);
  // let izm2 = max(iz - 2u, 0u);
  let izp1 = iz + 1u;
let izp2 = iz + 2u;
let izm1 = iz - 1u;
let izm2 = iz - 2u;


  let i_xp1 = idx3(ixp1,iy,iz);
  let i_xp2 = idx3(ixp2,iy,iz);
  let i_xm1 = idx3(ixm1,iy,iz);
  let i_xm2 = idx3(ixm2,iy,iz);

  let i_yp1 = idx3(ix,iyp1,iz);
  let i_yp2 = idx3(ix,iyp2,iz);
  let i_ym1 = idx3(ix,iym1,iz);
  let i_ym2 = idx3(ix,iym2,iz);

  let i_zp1 = idx3(ix,iy,izp1);
  let i_zp2 = idx3(ix,iy,izp2);
  let i_zm1 = idx3(ix,iy,izm1);
  let i_zm2 = idx3(ix,iy,izm2);

  let invdz12 = FD4.invdz12;             // = 1/(12 dz)
  let invdz2  = 6.0 * invdz12;        // or precompute from FD4.invdz12

  let du_dx = (-u[i_xp2] + 8.0*u[i_xp1] - 8.0*u[i_xm1] + u[i_xm2]) * FD4.invdx12;
  let dv_dy = (-v[i_yp2] + 8.0*v[i_yp1] - 8.0*v[i_ym1] + v[i_ym2]) * FD4.invdy12;
  var dw_dz: f32;
  // let dw_dz = (-w[i_zp2] + 8.0*w[i_zp1] - 8.0*w[i_zm1] + w[i_zm2]) * FD4.invdz12;
  
  // if (iz <= 1u) {
  //   // forward 2nd order: (-3 f0 + 4 f1 - f2) / (2 dz)
  //   dw_dz = (-3.0*w[id] + 4.0*w[i_zp1] - 1.0*w[i_zp2]) * invdz2;
  // } else if (iz >= nz - 2u) {
  //   // backward 2nd order: (3 f0 - 4 f-1 + f-2) / (2 dz)
  //   dw_dz = ( 3.0*w[id] - 4.0*w[i_zm1] + 1.0*w[i_zm2]) * invdz2;
  // } else {
  //   // interior 4th order
  //   dw_dz = (-w[i_zp2] + 8.0*w[i_zp1] - 8.0*w[i_zm1] + w[i_zm2]) * invdz12;
  // }

// if (iz == 0u) {
//   // forward 2nd order at wall
//   dw_dz = (-3.0*w[id] + 4.0*w[i_zp1] - w[i_zp2]) * invdz2;
// } else if (iz == 1u) {
//   // still forward 2nd order (uses iz, iz+1, iz+2)
//   dw_dz = (-3.0*w[id] + 4.0*w[i_zp1] - w[i_zp2]) * invdz2;
// } else if (iz == nz-2u) {
//   // backward 2nd order
//   dw_dz = (3.0*w[id] - 4.0*w[i_zm1] + w[i_zm2]) * invdz2;
// } else if (iz == nz-1u) {
//   dw_dz = (3.0*w[id] - 4.0*w[i_zm1] + w[i_zm2]) * invdz2;
// } else {
//   // interior 4th order
//   dw_dz = (-w[i_zp2] + 8.0*w[i_zp1] - 8.0*w[i_zm1] + w[i_zm2]) * invdz12;
// }

// z-derivative for CG operator: homogeneous Neumann at lids
  if (iz == 0u || iz == nz-1u) {
    // enforce ∂p/∂z = 0 at the rigid lids for the operator
    dw_dz = 0.0;
  } else if (iz == 1u) {
    // forward 2nd order
    dw_dz = (-3.0*w[id] + 4.0*w[i_zp1] - w[i_zp2]) * invdz2;
  } else if (iz == nz-2u) {
    // backward 2nd order
    dw_dz = (3.0*w[id] - 4.0*w[i_zm1] + w[i_zm2]) * invdz2;
  } else {
    // interior 4th order
    dw_dz = (-w[i_zp2] + 8.0*w[i_zp1] - 8.0*w[i_zm1] + w[i_zm2]) * invdz12;
  }


  // if (iz == 0u || iz == nz - 1u) {
  //   dw_dz = 0.0;  // Neumann BC for psi at rigid lid/floor
  // }

  outDiv[id] = du_dx + dv_dy + dw_dz;
}
`;

  // ---------- grad4 ----------
  const GRAD4_WGSL = /* wgsl */`
${FD4_COMMON}

@group(0) @binding(0) var<storage, read>       psi : array<f32>;
@group(0) @binding(1) var<storage, read_write> gx  : array<f32>;
@group(0) @binding(2) var<storage, read_write> gy  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gz  : array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= FD4.N) { return; }

  let nx = FD4.nx;
  let ny = FD4.ny;
  let ix = id % nx;
  let iy = (id / nx) % ny;
  let iz = id / (nx * ny);

  let ixp1 = wrap_inc(ix,nx);
  let ixp2 = wrap_inc(ixp1,nx);
  let ixm1 = wrap_dec(ix,nx);
  let ixm2 = wrap_dec(ixm1,nx);

  let iyp1 = wrap_inc(iy,ny);
  let iyp2 = wrap_inc(iyp1,ny);
  let iym1 = wrap_dec(iy,ny);
  let iym2 = wrap_dec(iym1,ny);

  let nz = FD4.nz;
  // let izp1 = wrap_inc(iz,nz);
  // let izp2 = wrap_inc(izp1,nz);
  // let izm1 = wrap_dec(iz,nz);
  // let izm2 = wrap_dec(izm1,nz);
  // let izp1 = min(iz + 1u, nz - 1u); 
  // let izp2 = min(iz + 2u, nz - 1u);
  // let izm1 = max(iz - 1u, 0u);
  // let izm2 = max(iz - 2u, 0u);
  let izp1 = iz + 1u;
let izp2 = iz + 2u;
let izm1 = iz - 1u;
let izm2 = iz - 2u;


  let i_xp1 = idx3(ixp1,iy,iz);
  let i_xp2 = idx3(ixp2,iy,iz);
  let i_xm1 = idx3(ixm1,iy,iz);
  let i_xm2 = idx3(ixm2,iy,iz);

  let i_yp1 = idx3(ix,iyp1,iz);
  let i_yp2 = idx3(ix,iyp2,iz);
  let i_ym1 = idx3(ix,iym1,iz);
  let i_ym2 = idx3(ix,iym2,iz);

  let i_zp1 = idx3(ix,iy,izp1);
  let i_zp2 = idx3(ix,iy,izp2);
  let i_zm1 = idx3(ix,iy,izm1);
  let i_zm2 = idx3(ix,iy,izm2);

  let dpsi_dx = (-psi[i_xp2] + 8.0*psi[i_xp1] - 8.0*psi[i_xm1] + psi[i_xm2]) * FD4.invdx12;
  let dpsi_dy = (-psi[i_yp2] + 8.0*psi[i_yp1] - 8.0*psi[i_ym1] + psi[i_ym2]) * FD4.invdy12;
  var dpsi_dz: f32;
  // let dpsi_dz = (-psi[i_zp2] + 8.0*psi[i_zp1] - 8.0*psi[i_zm1] + psi[i_zm2]) * FD4.invdz12;
  
  let invdz12 = FD4.invdz12;             // = 1/(12 dz)
  let invdz2  = 6.0 * invdz12;         // or precompute from FD4.invdz12

  // if (iz <= 1u) {
  //   // forward 2nd order: (-3 f0 + 4 f1 - f2) / (2 dz)
  //   dpsi_dz = (-3.0*psi[id] + 4.0*psi[i_zp1] - 1.0*psi[i_zp2]) * invdz2;
  // } else if (iz >= nz - 2u) {
  //   // backward 2nd order: (3 f0 - 4 f-1 + f-2) / (2 dz)
  //   dpsi_dz = ( 3.0*psi[id] - 4.0*psi[i_zm1] + 1.0*psi[i_zm2]) * invdz2;
  // } else {
  //   // interior 4th order
  //   dpsi_dz = (-psi[i_zp2] + 8.0*psi[i_zp1] - 8.0*psi[i_zm1] + psi[i_zm2]) * invdz12;
  // }

  // z-derivative for CG operator: homogeneous Neumann at lids
  if (iz == 0u || iz == nz-1u) {
    // enforce ∂p/∂z = 0 at the rigid lids for the operator
    dpsi_dz = 0.0;
  } else if (iz == 1u) {
    // forward 2nd order
    dpsi_dz = (-3.0*psi[id] + 4.0*psi[i_zp1] - psi[i_zp2]) * invdz2;
  } else if (iz == nz-2u) {
    // backward 2nd order
    dpsi_dz = (3.0*psi[id] - 4.0*psi[i_zm1] + psi[i_zm2]) * invdz2;
  } else {
    // interior 4th order
    dpsi_dz = (-psi[i_zp2] + 8.0*psi[i_zp1] - 8.0*psi[i_zm1] + psi[i_zm2]) * invdz12;
  }


  // if (iz == 0u || iz == nz - 1u) {
  //   dpsi_dz = 0.0;  // Neumann BC for psi at rigid lid/floor
  // }

  gx[id] = dpsi_dx;
  gy[id] = dpsi_dy;
  gz[id] = dpsi_dz;
}
`;
const GRAD4_BC_WGSL = /* wgsl */`
${FD4_COMMON}

@group(0) @binding(0) var<storage, read>       psi  : array<f32>;
@group(0) @binding(1) var<storage, read_write> gx   : array<f32>;
@group(0) @binding(2) var<storage, read_write> gy   : array<f32>;
@group(0) @binding(3) var<storage, read_write> gz   : array<f32>;
@group(0) @binding(5) var<storage, read>       rho0 : array<f32>;
@group(0) @binding(6) var<storage, read>       wstar: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= FD4.N) { return; }

  let nx = FD4.nx;
  let ny = FD4.ny;
  let ix = id % nx;
  let iy = (id / nx) % ny;
  let iz = id / (nx * ny);

  let ixp1 = wrap_inc(ix,nx);
  let ixp2 = wrap_inc(ixp1,nx);
  let ixm1 = wrap_dec(ix,nx);
  let ixm2 = wrap_dec(ixm1,nx);

  let iyp1 = wrap_inc(iy,ny);
  let iyp2 = wrap_inc(iyp1,ny);
  let iym1 = wrap_dec(iy,ny);
  let iym2 = wrap_dec(iym1,ny);

  let nz = FD4.nz;
  let izp1 = min(iz + 1u, nz - 1u); 
  let izp2 = min(iz + 2u, nz - 1u);
  let izm1 = max(iz - 1u, 0u);
  let izm2 = max(iz - 2u, 0u);

  let i_xp1 = idx3(ixp1,iy,iz);
  let i_xp2 = idx3(ixp2,iy,iz);
  let i_xm1 = idx3(ixm1,iy,iz);
  let i_xm2 = idx3(ixm2,iy,iz);

  let i_yp1 = idx3(ix,iyp1,iz);
  let i_yp2 = idx3(ix,iyp2,iz);
  let i_ym1 = idx3(ix,iym1,iz);
  let i_ym2 = idx3(ix,iym2,iz);

  let i_zp1 = idx3(ix,iy,izp1);
  let i_zp2 = idx3(ix,iy,izp2);
  let i_zm1 = idx3(ix,iy,izm1);
  let i_zm2 = idx3(ix,iy,izm2);

  let dpsi_dx = (-psi[i_xp2] + 8.0*psi[i_xp1] - 8.0*psi[i_xm1] + psi[i_xm2]) * FD4.invdx12;
  let dpsi_dy = (-psi[i_yp2] + 8.0*psi[i_yp1] - 8.0*psi[i_ym1] + psi[i_ym2]) * FD4.invdy12;

  let invdz12 = FD4.invdz12;
  let invdz2  = 6.0 * invdz12;

  var dpsi_dz: f32;

  // Final correction BC: ∂ψ/∂z = ρ0 w* at lids
  if (iz == 0u || iz == nz-1u) {
    dpsi_dz = rho0[id] * wstar[id];
  } else if (iz == 1u) {
    dpsi_dz = (-3.0*psi[id] + 4.0*psi[i_zp1] - psi[i_zp2]) * invdz2;
  } else if (iz == nz-2u) {
    dpsi_dz = (3.0*psi[id] - 4.0*psi[i_zm1] + psi[i_zm2]) * invdz2;
  } else {
    dpsi_dz = (-psi[i_zp2] + 8.0*psi[i_zp1] - 8.0*psi[i_zm1] + psi[i_zm2]) * invdz12;
  }

  gx[id] = dpsi_dx;
  gy[id] = dpsi_dy;
  gz[id] = dpsi_dz;
}
`;

 // ---------- div4 ----------
//   const DIV4_WGSL = /* wgsl */`
// ${FD4_COMMON}

// @group(0) @binding(0) var<storage, read>       u : array<f32>;
// @group(0) @binding(1) var<storage, read>       v : array<f32>;
// @group(0) @binding(2) var<storage, read>       w : array<f32>;
// @group(0) @binding(3) var<storage, read_write> outDiv : array<f32>;

// @compute @workgroup_size(${WG})
// fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
//   let id = gid.x;
//   if (id >= FD4.N) { return; }

//   let nx = FD4.nx;
//   let ny = FD4.ny;
//   let ix = id % nx;
//   let iy = (id / nx) % ny;
//   let iz = id / (nx * ny);

//   let ixp1 = wrap_inc(ix,nx);
//   let ixp2 = wrap_inc(ixp1,nx);
//   let ixm1 = wrap_dec(ix,nx);
//   let ixm2 = wrap_dec(ixm1,nx);

//   let iyp1 = wrap_inc(iy,ny);
//   let iyp2 = wrap_inc(iyp1,ny);
//   let iym1 = wrap_dec(iy,ny);
//   let iym2 = wrap_dec(iym1,ny);

//   let nz = FD4.nz;
//   let izp1 = wrap_inc(iz,nz);
//   let izp2 = wrap_inc(izp1,nz);
//   let izm1 = wrap_dec(iz,nz);
//   let izm2 = wrap_dec(izm1,nz);

//   let i_xp1 = idx3(ixp1,iy,iz);
//   let i_xp2 = idx3(ixp2,iy,iz);
//   let i_xm1 = idx3(ixm1,iy,iz);
//   let i_xm2 = idx3(ixm2,iy,iz);

//   let i_yp1 = idx3(ix,iyp1,iz);
//   let i_yp2 = idx3(ix,iyp2,iz);
//   let i_ym1 = idx3(ix,iym1,iz);
//   let i_ym2 = idx3(ix,iym2,iz);

//   let i_zp1 = idx3(ix,iy,izp1);
//   let i_zp2 = idx3(ix,iy,izp2);
//   let i_zm1 = idx3(ix,iy,izm1);
//   let i_zm2 = idx3(ix,iy,izm2);

//   let du_dx = (-u[i_xp2] + 8.0*u[i_xp1] - 8.0*u[i_xm1] + u[i_xm2]) * FD4.invdx12;
//   let dv_dy = (-v[i_yp2] + 8.0*v[i_yp1] - 8.0*v[i_ym1] + v[i_ym2]) * FD4.invdy12;
//   let dw_dz = (-w[i_zp2] + 8.0*w[i_zp1] - 8.0*w[i_zm1] + w[i_zm2]) * FD4.invdz12;

//   outDiv[id] = du_dx + dv_dy + dw_dz;
// }
// `;

//   // ---------- grad4 ----------
//   const GRAD4_WGSL = /* wgsl */`
// ${FD4_COMMON}

// @group(0) @binding(0) var<storage, read>       psi : array<f32>;
// @group(0) @binding(1) var<storage, read_write> gx  : array<f32>;
// @group(0) @binding(2) var<storage, read_write> gy  : array<f32>;
// @group(0) @binding(3) var<storage, read_write> gz  : array<f32>;

// @compute @workgroup_size(${WG})
// fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
//   let id = gid.x;
//   if (id >= FD4.N) { return; }

//   let nx = FD4.nx;
//   let ny = FD4.ny;
//   let ix = id % nx;
//   let iy = (id / nx) % ny;
//   let iz = id / (nx * ny);

//   let ixp1 = wrap_inc(ix,nx);
//   let ixp2 = wrap_inc(ixp1,nx);
//   let ixm1 = wrap_dec(ix,nx);
//   let ixm2 = wrap_dec(ixm1,nx);

//   let iyp1 = wrap_inc(iy,ny);
//   let iyp2 = wrap_inc(iyp1,ny);
//   let iym1 = wrap_dec(iy,ny);
//   let iym2 = wrap_dec(iym1,ny);

//   let nz = FD4.nz;
//   let izp1 = wrap_inc(iz,nz);
//   let izp2 = wrap_inc(izp1,nz);
//   let izm1 = wrap_dec(iz,nz);
//   let izm2 = wrap_dec(izm1,nz);

//   let i_xp1 = idx3(ixp1,iy,iz);
//   let i_xp2 = idx3(ixp2,iy,iz);
//   let i_xm1 = idx3(ixm1,iy,iz);
//   let i_xm2 = idx3(ixm2,iy,iz);

//   let i_yp1 = idx3(ix,iyp1,iz);
//   let i_yp2 = idx3(ix,iyp2,iz);
//   let i_ym1 = idx3(ix,iym1,iz);
//   let i_ym2 = idx3(ix,iym2,iz);

//   let i_zp1 = idx3(ix,iy,izp1);
//   let i_zp2 = idx3(ix,iy,izp2);
//   let i_zm1 = idx3(ix,iy,izm1);
//   let i_zm2 = idx3(ix,iy,izm2);

//   let dpsi_dx = (-psi[i_xp2] + 8.0*psi[i_xp1] - 8.0*psi[i_xm1] + psi[i_xm2]) * FD4.invdx12;
//   let dpsi_dy = (-psi[i_yp2] + 8.0*psi[i_yp1] - 8.0*psi[i_ym1] + psi[i_ym2]) * FD4.invdy12;
//   let dpsi_dz = (-psi[i_zp2] + 8.0*psi[i_zp1] - 8.0*psi[i_zm1] + psi[i_zm2]) * FD4.invdz12;

//   gx[id] = dpsi_dx;
//   gy[id] = dpsi_dy;
//   gz[id] = dpsi_dz;
// }
// `;

  // ---------- dot (stage 1) ----------
  const DOT_WGSL = /* wgsl */`
struct DotU {
  N: u32,
  numWG: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       a : array<f32>;
@group(0) @binding(1) var<storage, read>       b : array<f32>;
@group(0) @binding(2) var<storage, read_write> partials : array<f32>;
@group(0) @binding(3) var<uniform>             U : DotU;

var<workgroup> scratch : array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let id = gid.x;
  let lid0 = lid.x;

  var val = 0.0;
  if (id < U.N) {
    val = a[id] * b[id];
  }
  scratch[lid0] = val;
  workgroupBarrier();

  var stride = ${WG / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lid0 < stride) {
      scratch[lid0] = scratch[lid0] + scratch[lid0 + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid0 == 0u) {
    partials[wid.x] = scratch[0];
  }
}
`;

  // ---------- sum (stage 1) ----------
  const SUM_WGSL = /* wgsl */`
struct SumU {
  N: u32,
  numWG: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       a : array<f32>;
@group(0) @binding(1) var<storage, read_write> partials : array<f32>;
@group(0) @binding(2) var<uniform>             U : SumU;

var<workgroup> scratch : array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let id = gid.x;
  let lid0 = lid.x;

  var val = 0.0;
  if (id < U.N) {
    val = a[id];
  }
  scratch[lid0] = val;
  workgroupBarrier();

  var stride = ${WG / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lid0 < stride) {
      scratch[lid0] = scratch[lid0] + scratch[lid0 + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid0 == 0u) {
    partials[wid.x] = scratch[0];
  }
}
`;

  // ---------- reduce partials -> scalar ----------
  const REDUCE_WGSL = /* wgsl */`
struct ReduceU {
  numWG: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       partials : array<f32>;
@group(0) @binding(1) var<storage, read_write> scalar   : array<f32>;
@group(0) @binding(2) var<uniform>             U        : ReduceU;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  var s = 0.0;
  for (var i:u32 = 0u; i < U.numWG; i = i+1u) {
    s = s + partials[i];
  }
  scalar[0] = s;
}
`;

  // ---------- subtract mean (uses sumDiv / N) ----------
  const SUBTRACT_MEAN_WGSL = /* wgsl */`
struct MeanU {
  N: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> buf : array<f32>;
@group(0) @binding(1) var<storage, read>       sumDiv : array<f32>;
@group(0) @binding(2) var<uniform>             U   : MeanU;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }
  let mean = sumDiv[0] / f32(U.N);
  buf[i] = buf[i] - mean;
}
`;

  // ---------- clear buf = 0 ----------
  const CLEAR_WGSL = /* wgsl */`
struct ClearU {
  N: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> buf : array<f32>;
@group(0) @binding(1) var<uniform>             U   : ClearU;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }
  buf[i] = 0.0;
}
`;

  // ---------- copy dst = src ----------
  const COPY_WGSL = /* wgsl */`
struct CopyU {
  N: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             U   : CopyU;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }
  dst[i] = src[i];
}
`;

  // ---------- updatePsiR: psi += alpha*p, r -= alpha*Ap ----------
  const UPDATE_PSI_R_WGSL = /* wgsl */`
struct UpdateU {
  N: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> psi : array<f32>;
@group(0) @binding(1) var<storage, read_write> r   : array<f32>;
@group(0) @binding(2) var<storage, read>       p   : array<f32>;
@group(0) @binding(3) var<storage, read>       Ap  : array<f32>;
@group(0) @binding(4) var<storage, read>       rsoldBuf : array<f32>;
@group(0) @binding(5) var<storage, read>       pApBuf   : array<f32>;
@group(0) @binding(6) var<uniform>             U   : UpdateU;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }

  let rsold = rsoldBuf[0];
  let pAp   = pApBuf[0];

  // NaN checks that always work:
  if (pAp != pAp || rsold != rsold) { return; }
  
  // tiny denom check:
  if (abs(pAp) < 1e-20) { return; }

  let alpha = rsold / pAp;

  psi[i] = psi[i] + alpha * p[i];
  r[i]   = r[i]   - alpha * Ap[i];
}
`;

  // ---------- updateP: p = r + beta * p ----------
  const UPDATE_P_WGSL = /* wgsl */`
struct UpdatePU {
  N: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       r   : array<f32>;
@group(0) @binding(1) var<storage, read_write> p   : array<f32>;
@group(0) @binding(2) var<storage, read>       rsoldBuf : array<f32>;
@group(0) @binding(3) var<storage, read>       rsnewBuf : array<f32>;
@group(0) @binding(4) var<uniform>             U   : UpdatePU;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }

  let rsold = rsoldBuf[0];
  let rsnew = rsnewBuf[0];
  let beta  = rsnew / rsold;

  p[i] = r[i] + beta * p[i];
}
`;

  // ---------- copy scalar: dst[0] = src[0] ----------
  const COPY_SCALAR_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  dst[0] = src[0];
}
`;

  // ---------- final projection u -= gx, etc. ----------
  const PROJECT_SUB_WGSL = /* wgsl */`
struct ProjU {
  N: u32,
  _pad0: u32,   
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       gx : array<f32>;
@group(0) @binding(1) var<storage, read>       gy : array<f32>;
@group(0) @binding(2) var<storage, read>       gz : array<f32>;
@group(0) @binding(3) var<storage, read_write> u  : array<f32>;
@group(0) @binding(4) var<storage, read_write> v  : array<f32>;
@group(0) @binding(5) var<storage, read_write> w  : array<f32>;
@group(0) @binding(6) var<uniform>             U  : ProjU;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }
let s = 0.01; // or 0.001
u[i] = u[i] - s * gx[i];
v[i] = v[i] - s * gy[i];
w[i] = w[i] - s * gz[i];

}
`;

const CLAMP_W_WGSL = /* wgsl */`
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
`;

  const pipelines: Fd4Pipelines = {
    div4: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: DIV4_WGSL }), entryPoint: "main" },
      label: "fd4_div4",
    }),
    grad4: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: GRAD4_WGSL }), entryPoint: "main" },
      label: "fd4_grad4",
    }),
    dot: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: DOT_WGSL }), entryPoint: "main" },
      label: "fd4_dot",
    }),
    sum: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: SUM_WGSL }), entryPoint: "main" },
      label: "fd4_sum",
    }),
    reduce: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: REDUCE_WGSL }), entryPoint: "main" },
      label: "fd4_reduce",
    }),
    subtractMean: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: SUBTRACT_MEAN_WGSL }), entryPoint: "main" },
      label: "fd4_subtract_mean",
    }),
    clear: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: CLEAR_WGSL }), entryPoint: "main" },
      label: "fd4_clear",
    }),
    copy: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: COPY_WGSL }), entryPoint: "main" },
      label: "fd4_copy",
    }),
    updatePsiR: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: UPDATE_PSI_R_WGSL }), entryPoint: "main" },
      label: "fd4_updatePsiR",
    }),
    updateP: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: UPDATE_P_WGSL }), entryPoint: "main" },
      label: "fd4_updateP",
    }),
    copyScalar: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: COPY_SCALAR_WGSL }), entryPoint: "main" },
      label: "fd4_copyScalar",
    }),
    projectSubtract: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: PROJECT_SUB_WGSL }), entryPoint: "main" },
      label: "fd4_projectSubtract",
    }),
    mulCoeff: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: MUL_COEFF_WGSL }), entryPoint: "main" },
      label: "fd4_mulCoeff",
    }),
    mulCoeffInplace: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: MUL_COEFF_INPLACE_WGSL }), entryPoint: "main" },
      label: "fd4_mulCoeffInplace",
    }),
    clampW: device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: CLAMP_W_WGSL }), entryPoint: "main" },
      label: "fd4_clampW",
    }),
grad4_bc: device.createComputePipeline({
  layout: "auto",
  compute: { module: device.createShaderModule({ code: GRAD4_BC_WGSL }), entryPoint: "main" },
  label: "fd4_grad4_bc",
}),

  };

  // ---------- small helpers ----------

  const wgCount = numWG;

  function passDiv4(pass: GPUComputePassEncoder, u: GPUBuffer, v: GPUBuffer, w: GPUBuffer, outDiv: GPUBuffer) {
    const bg = device.createBindGroup({
      layout: pipelines.div4.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: u } },
        { binding: 1, resource: { buffer: v } },
        { binding: 2, resource: { buffer: w } },
        { binding: 3, resource: { buffer: outDiv } },
        { binding: 4, resource: { buffer: uniforms.fd4 } },
      ],
    });
    pass.setPipeline(pipelines.div4);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passGrad4(pass: GPUComputePassEncoder, psi: GPUBuffer, gx: GPUBuffer, gy: GPUBuffer, gz: GPUBuffer) {
    const bg = device.createBindGroup({
      layout: pipelines.grad4.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: psi } },
        { binding: 1, resource: { buffer: gx } },
        { binding: 2, resource: { buffer: gy } },
        { binding: 3, resource: { buffer: gz } },
        { binding: 4, resource: { buffer: uniforms.fd4 } },
      ],
    });
    pass.setPipeline(pipelines.grad4);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passDot(pass: GPUComputePassEncoder, a: GPUBuffer, b: GPUBuffer) {
    const bg = device.createBindGroup({
      layout: pipelines.dot.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: buffers.partials } },
        { binding: 3, resource: { buffer: uniforms.dotU } },
      ],
    });
    pass.setPipeline(pipelines.dot);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passSum(pass: GPUComputePassEncoder, a: GPUBuffer) {
    const bg = device.createBindGroup({
      layout: pipelines.sum.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: buffers.partials } },
        { binding: 2, resource: { buffer: uniforms.sumU } },
      ],
    });
    pass.setPipeline(pipelines.sum);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passReduce(pass: GPUComputePassEncoder, scalarBuf: GPUBuffer) {
    const bg = device.createBindGroup({
      layout: pipelines.reduce.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.partials } },
        { binding: 1, resource: { buffer: scalarBuf } },
        { binding: 2, resource: { buffer: uniforms.reduceU } },
      ],
    });
    pass.setPipeline(pipelines.reduce);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
  }

  function passSubtractMean(pass: GPUComputePassEncoder) {
    const bg = device.createBindGroup({
      layout: pipelines.subtractMean.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.bDiv } },
        { binding: 1, resource: { buffer: buffers.sumDiv } },
        { binding: 2, resource: { buffer: uniforms.meanU } },
      ],
    });
    pass.setPipeline(pipelines.subtractMean);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passClear(pass: GPUComputePassEncoder, buf: GPUBuffer) {
    const bg = device.createBindGroup({
      layout: pipelines.clear.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: { buffer: uniforms.clearU } },
      ],
    });
    pass.setPipeline(pipelines.clear);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passCopy(pass: GPUComputePassEncoder, src: GPUBuffer, dst: GPUBuffer) {
    const bg = device.createBindGroup({
      layout: pipelines.copy.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: uniforms.copyU } },
      ],
    });
    pass.setPipeline(pipelines.copy);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passMulCoeff(
    pass: GPUComputePassEncoder,
    coeff: GPUBuffer,
    src: GPUBuffer,
    dst: GPUBuffer
  ) {
    const bg = device.createBindGroup({
      layout: pipelines.mulCoeff.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: coeff } },
        { binding: 1, resource: { buffer: src } },
        { binding: 2, resource: { buffer: dst } },
        { binding: 3, resource: { buffer: uniforms.copyU } }, // layout: N only
      ],
    });
    pass.setPipeline(pipelines.mulCoeff);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passMulCoeffInplace(
    pass: GPUComputePassEncoder,
    coeff: GPUBuffer,
    buf: GPUBuffer
  ) {
    const bg = device.createBindGroup({
      layout: pipelines.mulCoeffInplace.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: coeff } },
        { binding: 1, resource: { buffer: buf } },
        { binding: 2, resource: { buffer: uniforms.copyU } }, // N-only uniform
      ],
    });
    pass.setPipeline(pipelines.mulCoeffInplace);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passUpdatePsiR(pass: GPUComputePassEncoder) {
    const bg = device.createBindGroup({
      layout: pipelines.updatePsiR.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.psi } },
        { binding: 1, resource: { buffer: buffers.r } },
        { binding: 2, resource: { buffer: buffers.p } },
        { binding: 3, resource: { buffer: buffers.Ap } },
        { binding: 4, resource: { buffer: buffers.rsold } },
        { binding: 5, resource: { buffer: buffers.pAp } },
        { binding: 6, resource: { buffer: uniforms.meanU } }, // reuse layout: N only
      ],
    });
    pass.setPipeline(pipelines.updatePsiR);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passUpdateP(pass: GPUComputePassEncoder) {
    const bg = device.createBindGroup({
      layout: pipelines.updateP.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.r } },
        { binding: 1, resource: { buffer: buffers.p } },
        { binding: 2, resource: { buffer: buffers.rsold } },
        { binding: 3, resource: { buffer: buffers.rsnew } },
        { binding: 4, resource: { buffer: uniforms.meanU } }, // reuse layout
      ],
    });
    pass.setPipeline(pipelines.updateP);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passCopyScalar(pass: GPUComputePassEncoder, src: GPUBuffer, dst: GPUBuffer) {
    const bg = device.createBindGroup({
      layout: pipelines.copyScalar.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
      ],
    });
    pass.setPipeline(pipelines.copyScalar);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
  }

  function passProjectSubtract(pass: GPUComputePassEncoder, u: GPUBuffer, v: GPUBuffer, w: GPUBuffer) {
    const bg = device.createBindGroup({
      layout: pipelines.projectSubtract.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.gx } },
        { binding: 1, resource: { buffer: buffers.gy } },
        { binding: 2, resource: { buffer: buffers.gz } },
        { binding: 3, resource: { buffer: u } },
        { binding: 4, resource: { buffer: v } },
        { binding: 5, resource: { buffer: w } },
        { binding: 6, resource: { buffer: uniforms.projU } },
      ],
    });
    pass.setPipeline(pipelines.projectSubtract);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passClampW(pass: GPUComputePassEncoder, w: GPUBuffer) {
    const bg = device.createBindGroup({
      layout: pipelines.clampW.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: w } },
        { binding: 1, resource: { buffer: uniforms.fd4 } },
      ],
    });
    pass.setPipeline(pipelines.clampW);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgCount);
  }

  function passGrad4BC(
  pass: GPUComputePassEncoder,
  psi: GPUBuffer,
  gx: GPUBuffer,
  gy: GPUBuffer,
  gz: GPUBuffer,
  rho0: GPUBuffer,
  wstar: GPUBuffer
) {
  const bg = device.createBindGroup({
    layout: pipelines.grad4_bc.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: psi } },
      { binding: 1, resource: { buffer: gx } },
      { binding: 2, resource: { buffer: gy } },
      { binding: 3, resource: { buffer: gz } },
      { binding: 4, resource: { buffer: uniforms.fd4 } }, // FD4 at binding(6) because 0..5 used
      { binding: 5, resource: { buffer: rho0 } },
      { binding: 6, resource: { buffer: wstar } },
    ],
  });
  pass.setPipeline(pipelines.grad4_bc);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(wgCount);
}


  function applyL4(pass: GPUComputePassEncoder, inv_rho0: GPUBuffer) {
    // gx,gy,gz = grad4(p)
    passGrad4(pass, buffers.p, buffers.gx, buffers.gy, buffers.gz);

    // scale by inv_rho0: gx,gy,gz <- inv_rho0 * g*
    passMulCoeffInplace(pass, inv_rho0, buffers.gx);
    passMulCoeffInplace(pass, inv_rho0, buffers.gy);
    passMulCoeffInplace(pass, inv_rho0, buffers.gz);


    // Ap = div4(inv_rho0*grad4(p))
    passDiv4(pass, buffers.gx, buffers.gy, buffers.gz, buffers.Ap);
  }


  // ---------- public synchronous project ----------
  function project(
    pass: GPUComputePassEncoder,
    u: GPUBuffer, v: GPUBuffer, w: GPUBuffer,
    rho0: GPUBuffer, inv_rho0: GPUBuffer,
    maxIter = 40
  ) {

    // TEMP: wipe velocities to zero to test projection pipeline
// passClear(pass, u);
// passClear(pass, v);
// passClear(pass, w);
// passClampW(pass, w);


    // 1) bDiv = div4(rho0*u*, rho0*v*, rho0*w*)
    // reuse gx/gy/gz as scratch for scaled velocities
    passMulCoeff(pass, rho0, u, buffers.gx);
    passMulCoeff(pass, rho0, v, buffers.gy);
    passMulCoeff(pass, rho0, w, buffers.gz);
    passDiv4(pass, buffers.gx, buffers.gy, buffers.gz, buffers.bDiv);


    // 2) subtract mean(bDiv) via GPU reduction
    passSum(pass, buffers.bDiv);   // partials

    passReduce(pass, buffers.sumDiv); // sumDiv[0] = sum

    passSubtractMean(pass);

    // 3) psi = 0; r = b; p = r
    passClear(pass, buffers.psi);
    passCopy(pass, buffers.bDiv, buffers.r);
    passCopy(pass, buffers.r, buffers.p);

    // 4) rsold = r·r
    passDot(pass, buffers.r, buffers.r);

    passReduce(pass, buffers.rsold);

    // 5) CG iterations (fixed maxIter)
    for (let k = 0; k < maxIter; k++) {
      // Ap = L4(p)
      applyL4(pass, inv_rho0);

      // pAp = p·Ap
      passDot(pass, buffers.p, buffers.Ap);

      passReduce(pass, buffers.pAp);

      // psi,r update using alpha
      passUpdatePsiR(pass);

      // rsnew = r·r
      passDot(pass, buffers.r, buffers.r);

      passReduce(pass, buffers.rsnew);

      // p = r + beta*p
      passUpdateP(pass);

      // rsold = rsnew (scalar copy)
      passCopyScalar(pass, buffers.rsnew, buffers.rsold);
    }

    // 6) compute grad4(psi)
    passGrad4(pass, buffers.psi, buffers.gx, buffers.gy, buffers.gz);
    // 6) compute grad(psi) with rigid-lid BC using w* and rho0
    // passGrad4BC(pass, buffers.psi, buffers.gx, buffers.gy, buffers.gz, rho0, w);


    // scale: g <- inv_rho0 * g
    passMulCoeffInplace(pass, inv_rho0, buffers.gx);
    passMulCoeffInplace(pass, inv_rho0, buffers.gy);
    passMulCoeffInplace(pass, inv_rho0, buffers.gz);


    // subtract scaled gradient from velocity
    passProjectSubtract(pass, u, v, w);

    // clamp w at top and bottom layer to enforce lid
    // passClampW(pass, w);
  }
  

  return {
    project,
    resources: { buffers, uniforms, pipelines, wgCount },
    debugReadbacks: { readbackPAp, readbackRsold, readbackRsnew },
  };
}
