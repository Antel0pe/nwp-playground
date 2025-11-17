// // project_velocity.ts
// import type { SimDims } from "../init_boussinesq";
// import { makeDivergence } from "./divergence";
// import { makeJacobiPoisson } from "./jacobi_poisson";
// import { makeGradSubtract } from "./grad_subtract";

// export function makeProjection(opts:{ device: GPUDevice; dims: SimDims }) {
//   const { device, dims } = opts;
//   const N = dims.nx * dims.ny * dims.nz;
//   const bytes = N * 4;

//   // scratch: div, psi ping-pong
//   const div = device.createBuffer({
//     size: bytes,
//     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
//   });

//   // psiA, psiB: start as zero (Jacobi from 0)
//   const psiA = device.createBuffer({
//     size: bytes,
//     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
//     mappedAtCreation: true,
//   });
//   {
//     const view = new Float32Array(psiA.getMappedRange());
//     view.fill(0);
//     psiA.unmap();
//   }

//   const psiB = device.createBuffer({
//     size: bytes,
//     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
//     mappedAtCreation: true,
//   });
//   {
//     const view = new Float32Array(psiB.getMappedRange());
//     view.fill(0);
//     psiB.unmap();
//   }

//   const divergence = makeDivergence({ device, dims });
//   const jacobi = makeJacobiPoisson({ device, dims });
//   const gradSub = makeGradSubtract({ device, dims });

//   function project(pass: GPUComputePassEncoder, u: GPUBuffer, v: GPUBuffer, w: GPUBuffer, iters = 40) {
//     // div = ∂u+∂v+∂w
//     divergence.dispatch(pass, { u, v, w, out_div: div });

//     // initialize psi = 0 (one clear)
//     // You already have a "CLEAR" utility. Bind it here once for psiA.
//     // Or just run a quick copy from a zero buffer you keep around.
//     // Minimal version: reuse your CLEAR pipeline outside this module.

//     // Jacobi iterations (ping-pong psiA <-> psiB)
//     var ping = psiA, pong = psiB;
//     for (var k=0; k<iters; k++) {
//       jacobi.dispatch(pass, { rhs_div: div, psi_in: ping, psi_out: pong });
//       let t = ping; ping = pong; pong = t;
//     }
//     // ping holds the latest solution

//     // velocity correction
//     gradSub.dispatch(pass, { psi: ping, u, v, w });
//   }

//   return { project, resources: { div, psiA, psiB } };
// }
// projection/project_velocity.ts

// projection/project_velocity.ts
//
// WebGPU version of the Python spectral projection:
//
//   u_corr, v_corr, w_corr, psi = project_velocity(u,v,w,dx,dy,dz)
//
// Here:
//   - u,v,w are real-space velocity buffers (f32) laid out [z][y][x].
//   - We use fft3d.forwardFFTN / inverseFFTN for transforms.
//   - psiReal is the real-space scalar potential (for debugging/visualization).

import type { SimDims } from "../init_boussinesq";
import { forwardFFTN, makeFFT3DContext } from "./fft3d";

export type ProjectionCtx = {
  project: (encoder: GPUCommandEncoder, u: GPUBuffer, v: GPUBuffer, w: GPUBuffer) => void;
  psiReal: GPUBuffer;
};

export function makeProjection(opts: {
  device: GPUDevice;
  dims: SimDims;
}): ProjectionCtx {
  const { device, dims } = opts;
  const { nx, ny, nz, dx, dy, dz } = dims;
  const N = nx * ny * nz;

  // --- 1) FFT context (3D) ----------------------------------------------

  const fftCtx = makeFFT3DContext(device, { nx, ny, nz });

  // --- 2) Precompute wavenumbers: _wavenumbers = 2π * fftfreq(n,d) -----

  function fftfreq(n: number, d: number): Float32Array {
    const out = new Float32Array(n);
    const val = 1.0 / (n * d);
    const N2 = Math.floor(n / 2);
    for (let i = 0; i < N2; ++i) {
      out[i] = i * val;
    }
    if (n % 2 === 0) {
      out[N2] = -N2 * val;
      for (let i = N2 + 1; i < n; ++i) {
        out[i] = (i - n) * val;
      }
    } else {
      for (let i = N2; i < n; ++i) {
        out[i] = (i - n) * val;
      }
    }
    return out;
  }

  function angularWavenumbers(n: number, d: number): Float32Array {
    const freq = fftfreq(n, d);
    const out = new Float32Array(n);
    const twoPi = 2.0 * Math.PI;
    for (let i = 0; i < n; ++i) {
      out[i] = twoPi * freq[i];
    }
    return out;
  }

  const kxArr = angularWavenumbers(nx, dx);
  const kyArr = angularWavenumbers(ny, dy);
  const kzArr = angularWavenumbers(nz, dz);

  const kxBuf = device.createBuffer({
    size: nx * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "proj_kx",
  });
  const kyBuf = device.createBuffer({
    size: ny * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "proj_ky",
  });
  const kzBuf = device.createBuffer({
    size: nz * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "proj_kz",
  });

  device.queue.writeBuffer(kxBuf, 0, kxArr);
  device.queue.writeBuffer(kyBuf, 0, kyArr);
  device.queue.writeBuffer(kzBuf, 0, kzArr);

  // --- 3) Spectral buffers U,V,W,psi_k ----------------------------------

  const complexSize = N * 2 * 4; // vec2<f32>
  const Uc = device.createBuffer({
    size: complexSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: "proj_Uc",
  });
  const Vc = device.createBuffer({
    size: complexSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: "proj_Vc",
  });
  const Wc = device.createBuffer({
    size: complexSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: "proj_Wc",
  });
  const psi_k = device.createBuffer({
    size: complexSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: "proj_psi_k",
  });

  const psiReal = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: "proj_psiReal",
  });

  const dimsBuf = device.createBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "proj_dims",
  });
  {
    const u32 = new Uint32Array([nx, ny, nz, 0]);
    device.queue.writeBuffer(dimsBuf, 0, u32);
  }

  // --- 4) K-space projection WGSL (Python port) -------------------------

  const module = device.createShaderModule({
    label: "project_velocity_kspace",
    code: /* wgsl */`
struct Dims {
  nx: u32,
  ny: u32,
  nz: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> U : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> V : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> W : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> PSI : array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> kx_arr : array<f32>;
@group(0) @binding(5) var<storage, read> ky_arr : array<f32>;
@group(0) @binding(6) var<storage, read> kz_arr : array<f32>;
@group(0) @binding(7) var<uniform> dims : Dims;

// layout: idx = iz*(ny*nx) + iy*nx + ix
fn idx_to_ijk(idx: u32, nx: u32, ny: u32) -> vec3<u32> {
  let plane = ny * nx;
  let iz = idx / plane;
  let rem = idx - iz * plane;
  let iy = rem / nx;
  let ix = rem - iy * nx;
  return vec3<u32>(ix, iy, iz);
}

fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x + b.x, a.y + b.y);
}

fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x - b.x, a.y - b.y);
}

fn c_mul_real(a: vec2<f32>, s: f32) -> vec2<f32> {
  return vec2<f32>(a.x * s, a.y * s);
}

fn c_mul_i(a: vec2<f32>) -> vec2<f32> {
  // i * (a.x + i a.y) = -a.y + i a.x
  return vec2<f32>(-a.y, a.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nx = dims.nx;
  let ny = dims.ny;
  let nz = dims.nz;
  let N: u32 = nx * ny * nz;

  let idx = gid.x;
  if (idx >= N) { return; }

  let ijk = idx_to_ijk(idx, nx, ny);
  let ix = ijk.x;
  let iy = ijk.y;
  let iz = ijk.z;

  let kx = kx_arr[ix];
  let ky = ky_arr[iy];
  let kz = kz_arr[iz];

  let k2 = kx*kx + ky*ky + kz*kz;

  let Uk = U[idx];
  let Vk = V[idx];
  let Wk = W[idx];

  // div_k = i*(kx*U + ky*V + kz*W)
  let kxU = c_mul_real(Uk, kx);
  let kyV = c_mul_real(Vk, ky);
  let kzW = c_mul_real(Wk, kz);

  var div_k = c_add(kxU, c_add(kyV, kzW));
  div_k = c_mul_i(div_k);

  var psi_k = vec2<f32>(0.0, 0.0);
  if (k2 != 0.0) {
    let invk2 = -1.0 / k2;
    psi_k = c_mul_real(div_k, invk2);
  } else {
    psi_k = vec2<f32>(0.0, 0.0);
  }

  PSI[idx] = psi_k;

  // gradient: Gx_k = i*kx*psi_k, etc.
  let Gx_k = c_mul_i(c_mul_real(psi_k, kx));
  let Gy_k = c_mul_i(c_mul_real(psi_k, ky));
  let Gz_k = c_mul_i(c_mul_real(psi_k, kz));

  U[idx] = c_sub(Uk, Gx_k);
  V[idx] = c_sub(Vk, Gy_k);
  W[idx] = c_sub(Wk, Gz_k);
}
`,
  });

  const bgl = device.createBindGroupLayout({
    label: "proj_kspace_bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });

  const pipeline = device.createComputePipeline({
    label: "proj_kspace_pipeline",
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: "main" },
  });

  const bind = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: Uc } },
      { binding: 1, resource: { buffer: Vc } },
      { binding: 2, resource: { buffer: Wc } },
      { binding: 3, resource: { buffer: psi_k } },
      { binding: 4, resource: { buffer: kxBuf } },
      { binding: 5, resource: { buffer: kyBuf } },
      { binding: 6, resource: { buffer: kzBuf } },
      { binding: 7, resource: { buffer: dimsBuf } },
    ],
  });

  const wg = 256;
  const groups = Math.ceil(N / wg);

  // --- 5) project(): match Python project_velocity -----------------------

  function project(encoder: GPUCommandEncoder, u: GPUBuffer, v: GPUBuffer, w: GPUBuffer) {
    // 1) forward FFTs: U,V,W
    forwardFFTN(encoder, fftCtx, u, Uc);
    forwardFFTN(encoder, fftCtx, v, Vc);
    forwardFFTN(encoder, fftCtx, w, Wc);

    // 2) k-space projection kernel
    {
      const pass = encoder.beginComputePass({ label: "proj_kspace_pass" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(groups, 1, 1);
      pass.end();
    }

    // 3) inverse FFTs: corrected velocities and psi
    inverseFFTN(encoder, fftCtx, Uc, u);
    inverseFFTN(encoder, fftCtx, Vc, v);
    inverseFFTN(encoder, fftCtx, Wc, w);
    inverseFFTN(encoder, fftCtx, psi_k, psiReal);
  }

  return { project, psiReal };
}
