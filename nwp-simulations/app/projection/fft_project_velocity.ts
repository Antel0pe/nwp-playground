// fft_project_velocity.ts
export type SimDims = {
  nx: number;
  ny: number;
  nz: number;
  dx: number;
  dy: number;
  dz: number;
};

type FftProjectionPipelines = {
  packRealToComplex: GPUComputePipeline;
  unpackComplexToReal: GPUComputePipeline;
  fftX: GPUComputePipeline;
  fftY: GPUComputePipeline;
  fftZ: GPUComputePipeline;
  projectSpectral: GPUComputePipeline;
  normalizeInverse: GPUComputePipeline;
};

type FftProjectionResources = {
  U: GPUBuffer;  // complex
  V: GPUBuffer;
  W: GPUBuffer;
  psiK: GPUBuffer;
  scratch: GPUBuffer; // for FFT ping-pong
  kx: GPUBuffer;
  ky: GPUBuffer;
  kz: GPUBuffer;
  dims: GPUBuffer;     // uniform: nx,ny,nz,N
  fftParams: GPUBuffer; // uniform: len, axis, sign
};

const FFT_WGSL = /* wgsl */`
struct Dims {
  nx: u32;
  ny: u32;
  nz: u32;
  N:  u32;
};

struct FftParams {
  len: f32;   // length of current 1D FFT (nx, ny, or nz)
  axis: f32;  // 0 = x, 1 = y, 2 = z
  sign: f32;  // +1 for forward, -1 for inverse
  _pad: f32;  // padding
};

@group(0) @binding(0) var<storage, read_write> realIn : array<f32>;

@group(0) @binding(1) var<storage, read_write> complexOut : array<vec2<f32>>;
@group(0) @binding(2) var<uniform> dimsBuf : Dims;


@group(1) @binding(0) var<storage, read>  complexIn  : array<vec2<f32>>;
@group(1) @binding(1) var<storage, read_write> complexScratch : array<vec2<f32>>;
@group(1) @binding(2) var<uniform>  fftParams : FftParams;

@group(2) @binding(0) var<storage, read_write> complexFieldU : array<vec2<f32>>;
@group(2) @binding(1) var<storage, read_write> complexFieldV : array<vec2<f32>>;
@group(2) @binding(2) var<storage, read_write> complexFieldW : array<vec2<f32>>;
@group(2) @binding(3) var<storage, read_write> psiK : array<vec2<f32>>;
@group(2) @binding(4) var<storage, read> kxBuf : array<f32>;
@group(2) @binding(5) var<storage, read> kyBuf : array<f32>;
@group(2) @binding(6) var<storage, read> kzBuf : array<f32>;

// ---------- helpers ----------

fn linear_index(i:u32, j:u32, k:u32, nx:u32, ny:u32, nz:u32) -> u32 {
  return i + nx * (j + ny * k);
}

// complex multiply
fn cmul(a:vec2<f32>, b:vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x*b.x - a.y*b.y,
    a.x*b.y + a.y*b.x
  );
}

// complex add
fn cadd(a:vec2<f32>, b:vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x + b.x, a.y + b.y);
}

// complex scale
fn cscale(a:vec2<f32>, s:f32) -> vec2<f32> {
  return vec2<f32>(a.x * s, a.y * s);
}

// exp(i * sign * 2π * m * n / len)
fn twiddle(m:u32, n:u32, len:u32, sign:f32) -> vec2<f32> {
  let two_pi = 6.283185307179586;
  let theta = sign * two_pi * f32(m) * f32(n) / f32(len);
  return vec2<f32>(cos(theta), sin(theta));
}

// ---------- kernel: pack real -> complex (imag = 0) ----------

@compute @workgroup_size(256)
fn pack_real_to_complex(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= dimsBuf.N) { return; }
  let v = realIn[idx];
  complexOut[idx] = vec2<f32>(v, 0.0);
}

// ---------- kernel: naive 1D DFT along one axis ----------
//
// Each invocation handles ONE line:
//   axis 0: line id = j + ny * k  (len = nx)
//   axis 1: line id = i + nx * k  (len = ny)
//   axis 2: line id = i + nx * j  (len = nz)

@compute @workgroup_size(64)
fn fft1d_naive(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lineId = gid.x;

  let nx = dimsBuf.nx;
  let ny = dimsBuf.ny;
  let nz = dimsBuf.nz;
let len  = u32(fftParams.len + 0.5);
let axis = u32(fftParams.axis + 0.5);
let sign = fftParams.sign;

  var i0:u32 = 0u;
  var j0:u32 = 0u;
  var k0:u32 = 0u;

  // number of lines and mapping from lineId -> (i0,j0,k0) depends on axis
  if (axis == 0u) {
    // x-lines: j,k vary; i runs 0..nx-1
    let numLines = ny * nz;
    if (lineId >= numLines) { return; }
    j0 = lineId % ny;
    k0 = lineId / ny;
  } else if (axis == 1u) {
    // y-lines: i,k vary; j runs 0..ny-1
    let numLines = nx * nz;
    if (lineId >= numLines) { return; }
    i0 = lineId % nx;
    k0 = lineId / nx;
  } else {
    // z-lines: i,j vary; k runs 0..nz-1
    let numLines = nx * ny;
    if (lineId >= numLines) { return; }
    i0 = lineId % nx;
    j0 = lineId / nx;
  }

  // For each output mode m in 0..len-1, do sum over n
  for (var m:u32 = 0u; m < len; m = m + 1u) {
    var sum:vec2<f32> = vec2<f32>(0.0, 0.0);
    for (var n:u32 = 0u; n < len; n = n + 1u) {
      var i:u32 = 0u;
      var j:u32 = 0u;
      var k:u32 = 0u;

      if (axis == 0u) {
        i = n;
        j = j0;
        k = k0;
      } else if (axis == 1u) {
        i = i0;
        j = n;
        k = k0;
      } else {
        i = i0;
        j = j0;
        k = n;
      }

      let idx = linear_index(i,j,k,nx,ny,nz);
      let x = complexIn[idx];
      let w = twiddle(m, n, len, sign);
      sum = cadd(sum, cmul(x, w));
    }

    var i_out:u32 = 0u;
    var j_out:u32 = 0u;
    var k_out:u32 = 0u;

    if (axis == 0u) {
      i_out = m;
      j_out = j0;
      k_out = k0;
    } else if (axis == 1u) {
      i_out = i0;
      j_out = m;
      k_out = k0;
    } else {
      i_out = i0;
      j_out = j0;
      k_out = m;
    }

    let idx_out = linear_index(i_out, j_out, k_out, nx, ny, nz);
    complexScratch[idx_out] = sum;
  }
}

// ---------- kernel: copy scratch -> field (for ping-pong) + (optional) 1/N normalization ----------

@compute @workgroup_size(256)
fn normalize_and_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= dimsBuf.N) { return; }

  // For inverse FFT we divide by N, for forward we leave as-is.
  let len = f32(dimsBuf.N);
let sign = fftParams.sign;

  var val = complexScratch[idx];

  // sign < 0 => inverse transform
  if (sign < 0.0) {
    val = cscale(val, 1.0 / len);
  }

  complexFieldU[idx] = val; // we'll bind whatever field as complexFieldU when we use this
}

// ---------- kernel: spectral projection (U,V,W,psiK in-place) ----------

@compute @workgroup_size(256)
fn project_spectral(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= dimsBuf.N) { return; }

  let nx = dimsBuf.nx;
  let ny = dimsBuf.ny;
  let nz = dimsBuf.nz;

  let Nxy = nx * ny;
  let k = idx / Nxy;
  let rem = idx - k * Nxy;
  let j = rem / nx;
  let i = rem - j * nx;

  let ki = kxBuf[i];
  let kj = kyBuf[j];
  let kk = kzBuf[k];

  let Uc = complexFieldU[idx];
  let Vc = complexFieldV[idx];
  let Wc = complexFieldW[idx];

  // div_k = i * (kx U + ky V + kz W)
  let kU = cscale(Uc, ki);
  let kV = cscale(Vc, kj);
  let kW = cscale(Wc, kk);

  var sumKV = cadd(cadd(kU, kV), kW);

  let i_factor = vec2<f32>(0.0, 1.0);
  let div_k = cmul(i_factor, sumKV);

  let k2 = ki*ki + kj*kj + kk*kk;

  var psi_val:vec2<f32> = vec2<f32>(0.0, 0.0);
  if (k2 > 0.0) {
    psi_val = cscale(div_k, -1.0 / k2);
  }
  psiK[idx] = psi_val;

  // Gx_k = i * kx * psi_k
  let psi_kx = cscale(psi_val, ki);
  let psi_ky = cscale(psi_val, kj);
  let psi_kz = cscale(psi_val, kk);

  let Gx_k = cmul(i_factor, psi_kx);
  let Gy_k = cmul(i_factor, psi_ky);
  let Gz_k = cmul(i_factor, psi_kz);

  complexFieldU[idx] = cadd(Uc, cscale(Gx_k, -1.0));
  complexFieldV[idx] = cadd(Vc, cscale(Gy_k, -1.0));
  complexFieldW[idx] = cadd(Wc, cscale(Gz_k, -1.0));
}

// ---------- kernel: unpack complex real-part -> real out ----------

@compute @workgroup_size(256)
fn unpack_complex_to_real(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= dimsBuf.N) { return; }
  realIn[idx] = complexOut[idx].x; // same binding trick as normalize_and_copy
}
`;

// -------------------------------------------------------------------

function makeWavenumbers(n: number, d: number): Float32Array {
  const out = new Float32Array(n);
  const twoPi = 2 * Math.PI;
  const nd = n * d;
  const half = Math.floor(n / 2);
  for (let k = 0; k < n; k++) {
    let kk = k <= half ? k : k - n;
    out[k] = twoPi * (kk / nd);
  }
  return out;
}

function createPipelines(device: GPUDevice): FftProjectionPipelines {
  const module = device.createShaderModule({ code: FFT_WGSL });

  const packRealToComplex = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "pack_real_to_complex" },
  });

  const fftX = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "fft1d_naive" },
  });

  const fftY = fftX; // same entry; axis controlled by uniform
  const fftZ = fftX;

  const normalizeInverse = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "normalize_and_copy" },
  });

  const projectSpectral = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "project_spectral" },
  });

  const unpackComplexToReal = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "unpack_complex_to_real" },
  });

  return {
    packRealToComplex,
    fftX,
    fftY,
    fftZ,
    projectSpectral,
    normalizeInverse,
    unpackComplexToReal,
  };
}

function createResources(device: GPUDevice, dims: SimDims): FftProjectionResources {
  const { nx, ny, nz, dx, dy, dz } = dims;
  const N = nx * ny * nz;

  const complexSize = N * 2 * 4;

  const makeComplex = () =>
    device.createBuffer({
      size: complexSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

  const U = makeComplex();
  const V = makeComplex();
  const W = makeComplex();
  const psiK = makeComplex();
  const scratch = makeComplex();

  const kxArr = makeWavenumbers(nx, dx);
  const kyArr = makeWavenumbers(ny, dy);
  const kzArr = makeWavenumbers(nz, dz);

  const kx = device.createBuffer({
    size: kxArr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const ky = device.createBuffer({
    size: kyArr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const kz = device.createBuffer({
    size: kzArr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

device.queue.writeBuffer(kx, 0, kxArr.buffer, 0, kxArr.byteLength);
device.queue.writeBuffer(ky, 0, kyArr.buffer, 0, kyArr.byteLength);
device.queue.writeBuffer(kz, 0, kzArr.buffer, 0, kzArr.byteLength);


  const dimsBuf = device.createBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const dimsArr = new Uint32Array([nx, ny, nz, N]);
  device.queue.writeBuffer(dimsBuf, 0, dimsArr);

const fftParams = device.createBuffer({
  size: 4 * 4, // len, axis, sign, pad
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

  return { U, V, W, psiK, scratch, kx, ky, kz, dims: dimsBuf, fftParams };
}

export function makeFftProjectVelocity(
  device: GPUDevice,
  dims: SimDims,
) {
  const pipelines = createPipelines(device);
  const res = createResources(device, dims);
  const { nx, ny, nz } = dims;
  const N = nx * ny * nz;

  const workgroups1D = (n: number, wg: number) => Math.ceil(n / wg);

  function recordPack(
    encoder: GPUCommandEncoder,
    realBuf: GPUBuffer,
    complexBuf: GPUBuffer,
  ) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.packRealToComplex);
    const bind = device.createBindGroup({
      layout: pipelines.packRealToComplex.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: realBuf } },
        { binding: 1, resource: { buffer: complexBuf } },
        { binding: 2, resource: { buffer: res.dims } },
      ],
    });
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(workgroups1D(N, 256));
    pass.end();
  }

  function recordUnpack(
    encoder: GPUCommandEncoder,
    complexBuf: GPUBuffer,
    realBuf: GPUBuffer,
  ) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.unpackComplexToReal);
    const bind = device.createBindGroup({
      layout: pipelines.unpackComplexToReal.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: realBuf } },    // realIn
        { binding: 1, resource: { buffer: complexBuf } }, // complexOut
        { binding: 2, resource: { buffer: res.dims } },
      ],
    });
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(workgroups1D(N, 256));
    pass.end();
  }
  function writeFftParams(len: number, axis: 0 | 1 | 2, sign: 1 | -1) {
  const params = new Float32Array([len, axis, sign, 0.0]);
  device.queue.writeBuffer(res.fftParams, 0, params);
}

  

  function recordFftAxis(
    encoder: GPUCommandEncoder,
    fieldComplex: GPUBuffer,
    len: number,
    axis: 0 | 1 | 2,
    sign: 1 | -1,
  ) {
    // 1) fft1d_naive: complexIn -> scratch
{
  const numLines =
    axis === 0 ? ny * nz :
    axis === 1 ? nx * nz :
                 nx * ny;

  writeFftParams(len, axis, sign);

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipelines.fftX);

  // group(0): dimsBuf at binding 2
  const bg0 = device.createBindGroup({
    layout: pipelines.fftX.getBindGroupLayout(0),
    entries: [
      { binding: 2, resource: { buffer: res.dims } },
    ],
  });

  // group(1): complexIn, complexScratch, fftParams
  const bg1 = device.createBindGroup({
    layout: pipelines.fftX.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: fieldComplex } }, // complexIn
      { binding: 1, resource: { buffer: res.scratch } },  // complexScratch
      { binding: 2, resource: { buffer: res.fftParams } },// fftParams
    ],
  });

  pass.setBindGroup(0, bg0);
  pass.setBindGroup(1, bg1);
  pass.dispatchWorkgroups(workgroups1D(numLines, 64));
  pass.end();
}


    {
  writeFftParams(len, axis, sign);

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipelines.normalizeInverse);

  const bg0 = device.createBindGroup({
    layout: pipelines.normalizeInverse.getBindGroupLayout(0),
    entries: [
      { binding: 2, resource: { buffer: res.dims } },
    ],
  });

  const bg1 = device.createBindGroup({
    layout: pipelines.normalizeInverse.getBindGroupLayout(1),
    entries: [
      { binding: 1, resource: { buffer: res.scratch } },   // complexScratch
      { binding: 2, resource: { buffer: res.fftParams } }, // fftParams
    ],
  });

  const bg2 = device.createBindGroup({
    layout: pipelines.normalizeInverse.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: fieldComplex } },  // complexFieldU
    ],
  });

  pass.setBindGroup(0, bg0);
  pass.setBindGroup(1, bg1);
  pass.setBindGroup(2, bg2);
  pass.dispatchWorkgroups(workgroups1D(N, 256));
  pass.end();
}

  }

  function recordProjectSpectral(encoder: GPUCommandEncoder) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipelines.projectSpectral);

  const bg0 = device.createBindGroup({
    layout: pipelines.projectSpectral.getBindGroupLayout(0),
    entries: [
      { binding: 2, resource: { buffer: res.dims } },
    ],
  });

  const bg2 = device.createBindGroup({
    layout: pipelines.projectSpectral.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: res.U } },
      { binding: 1, resource: { buffer: res.V } },
      { binding: 2, resource: { buffer: res.W } },
      { binding: 3, resource: { buffer: res.psiK } },
      { binding: 4, resource: { buffer: res.kx } },
      { binding: 5, resource: { buffer: res.ky } },
      { binding: 6, resource: { buffer: res.kz } },
    ],
  });

  pass.setBindGroup(0, bg0);
  pass.setBindGroup(2, bg2);
  pass.dispatchWorkgroups(workgroups1D(N, 256));
  pass.end();
}

  return {
    /**
     * Project u,v,w in-place using FFT-based pressure solve, periodic, like the Python code.
     * uBuf, vBuf, wBuf: real-valued velocity fields (Float32, length N).
     */
    project(encoder: GPUCommandEncoder, uBuf: GPUBuffer, vBuf: GPUBuffer, wBuf: GPUBuffer) {
      // 1) pack real -> complex
      recordPack(encoder, uBuf, res.U);
      recordPack(encoder, vBuf, res.V);
      recordPack(encoder, wBuf, res.W);

      // 2) forward FFT (axis 0,1,2) for each velocity component
      // forward sign = +1
      recordFftAxis(encoder, res.U, nx, 0, +1);
      recordFftAxis(encoder, res.U, ny, 1, +1);
      recordFftAxis(encoder, res.U, nz, 2, +1);

      recordFftAxis(encoder, res.V, nx, 0, +1);
      recordFftAxis(encoder, res.V, ny, 1, +1);
      recordFftAxis(encoder, res.V, nz, 2, +1);

      recordFftAxis(encoder, res.W, nx, 0, +1);
      recordFftAxis(encoder, res.W, ny, 1, +1);
      recordFftAxis(encoder, res.W, nz, 2, +1);

      // 3) spectral projection
      recordProjectSpectral(encoder);

      // 4) inverse FFT on corrected fields (sign = -1, with normalization)
      recordFftAxis(encoder, res.U, nx, 0, -1);
      recordFftAxis(encoder, res.U, ny, 1, -1);
      recordFftAxis(encoder, res.U, nz, 2, -1);

      recordFftAxis(encoder, res.V, nx, 0, -1);
      recordFftAxis(encoder, res.V, ny, 1, -1);
      recordFftAxis(encoder, res.V, nz, 2, -1);

      recordFftAxis(encoder, res.W, nx, 0, -1);
      recordFftAxis(encoder, res.W, ny, 1, -1);
      recordFftAxis(encoder, res.W, nz, 2, -1);

      // 5) unpack complex-real back into u,v,w
      recordUnpack(encoder, res.U, uBuf);
      recordUnpack(encoder, res.V, vBuf);
      recordUnpack(encoder, res.W, wBuf);
    },

    // If you want psi in physical space, you can inverse-FFT psiK similarly.
    resources: res,
  };
}
