// fft1d_real.ts
//
// 1D FFT on real input, batched.
// - Real input (f32) → complex buffer (vec2<f32>) via bit-reversal
// - In-place Cooley–Tukey DIT FFT on complex buffer
// - Optional inverse transform back to real (with 1/N scaling)
//
// This is intended as a building block for your FFT-based projection.
// You can create one plan per dimension (x, y, z) and per field (u,v,w)
// or reuse the same plan for multiple buffers with the same shape.

export type FFT1DRealPlan = {
  device: GPUDevice;
  n: number;           // FFT length (power of two)
  batches: number;     // number of independent lines
  logN: number;

  // Working complex buffer: size = n * batches * sizeof(vec2<f32>)
  complexBuf: GPUBuffer;

  // Uniform buffers
  bitrevUniformBuf: GPUBuffer;
  fftUniformBuf: GPUBuffer;
  scaleUniformBuf: GPUBuffer;

  // Pipelines
  bitrevPipeline: GPUComputePipeline;
  fftStagePipeline: GPUComputePipeline;
  scalePipeline: GPUComputePipeline;

  // Bind group layouts (shared)
  bitrevBGL: GPUBindGroupLayout;
  fftBGL: GPUBindGroupLayout;
  scaleBGL: GPUBindGroupLayout;
};

const WG_SIZE_X = 256;

// --- WGSL kernels ---------------------------------------------------------

const BITREV_WGSL = /* wgsl */`
struct BitrevUniforms {
  n: u32;
  logN: u32;
  batchStrideReal: u32;
  batchStrideComplex: u32;
};

@group(0) @binding(0) var<storage, read>  inReal  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outC : array<vec2<f32>>;
@group(0) @binding(2) var<uniform> U : BitrevUniforms;

fn bit_reverse(x: u32, bits: u32) -> u32 {
  var y: u32 = 0u;
  var i: u32 = 0u;
  loop {
    if (i >= bits) { break; }
    let bit = (x >> i) & 1u;
    y = (y << 1u) | bit;
    i = i + 1u;
  }
  return y;
}

@compute @workgroup_size(${WG_SIZE_X}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let batch = gid.y;
  if (idx >= U.n) {
    return;
  }

  let srcBase = batch * U.batchStrideReal;
  let dstBase = batch * U.batchStrideComplex;

  let srcIdx = srcBase + idx;
  let revIdx = bit_reverse(idx, U.logN);
  let dstIdx = dstBase + revIdx;

  let x = inReal[srcIdx];
  outC[dstIdx] = vec2<f32>(x, 0.0);
}
`;

const FFT_STAGE_WGSL = /* wgsl */`
struct FFTUniforms {
  n: u32;
  stage: u32;       // 1..logN
  direction: i32;   // +1 forward, -1 inverse
  batchStride: u32; // stride (in elements) between batches in 'data'
  _pad: u32;
};

@group(0) @binding(0) var<storage, read_write> data : array<vec2<f32>>;
@group(0) @binding(1) var<uniform> U : FFTUniforms;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

const PI: f32 = 3.141592653589793;

@compute @workgroup_size(${WG_SIZE_X}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = U.n;
  let numButterflies = n / 2u;

  let bflyIndex = gid.x;
  let batch = gid.y;

  if (bflyIndex >= numButterflies) {
    return;
  }

  // Cooley–Tukey DIT in-place butterfly indexing
  let halfM = 1u << (U.stage - 1u);
  let m = halfM << 1u;

  let j = bflyIndex & (halfM - 1u);  // position within this m-sized group
  let segment = bflyIndex / halfM;   // which group
  let k = segment * m;

  let base = batch * U.batchStride;
  let i0 = base + k + j;
  let i1 = i0 + halfM;

  let u = data[i0];
  let v = data[i1];

  // Twiddle factor: exp(-i * dir * 2π * j / m)
  let dir = f32(U.direction);
  let angle = dir * -2.0 * PI * f32(j) / f32(m);
  let twiddle = vec2<f32>(cos(angle), sin(angle));

  let t = complex_mul(v, twiddle);

  data[i0] = u + t;
  data[i1] = u - t;
}
`;

const SCALE_INV_WGSL = /* wgsl */`
struct ScaleUniforms {
  n: u32;
  batchStrideComplex: u32;
  batchStrideReal: u32;
};

@group(0) @binding(0) var<storage, read>      inC  : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outR : array<f32>;
@group(0) @binding(2) var<uniform> U : ScaleUniforms;

@compute @workgroup_size(${WG_SIZE_X}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let batch = gid.y;
  if (idx >= U.n) {
    return;
  }

  let baseC = batch * U.batchStrideComplex;
  let baseR = batch * U.batchStrideReal;

  let cIdx = baseC + idx;
  let rIdx = baseR + idx;

  let c = inC[cIdx];
  let scale = 1.0 / f32(U.n);

  // Standard inverse FFT: divide by N, take real part
  outR[rIdx] = c.x * scale;
}
`;

// --- Helper to assert power-of-two ----------------------------------------

function assertPowerOfTwo(n: number) {
  if (n <= 0 || (n & (n - 1)) !== 0) {
    throw new Error(`FFT length n must be power-of-two, got ${n}`);
  }
}

// --- Plan creation --------------------------------------------------------

export function createFFT1DRealPlan(opts: {
  device: GPUDevice;
  n: number;        // FFT length (power-of-two)
  batches: number;  // how many independent lines of length n
  label?: string;
}): FFT1DRealPlan {
  const { device, n, batches, label } = opts;
  assertPowerOfTwo(n);

  const logN = Math.round(Math.log2(n));
  const Ncomplex = n * batches;

  const complexBuf = device.createBuffer({
    size: Ncomplex * 2 * 4, // vec2<f32> -> 2 * 4 bytes
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: (label ?? "fft1d") + "_complexBuf",
  });

  const bitrevUniformBuf = device.createBuffer({
    size: 4 * 4, // n, logN, batchStrideReal, batchStrideComplex
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: (label ?? "fft1d") + "_bitrevUniforms",
  });

  const fftUniformBuf = device.createBuffer({
    size: 4 * 4, // n, stage, direction, batchStride, _pad
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: (label ?? "fft1d") + "_fftUniforms",
  });

  const scaleUniformBuf = device.createBuffer({
    size: 3 * 4, // n, batchStrideComplex, batchStrideReal
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: (label ?? "fft1d") + "_scaleUniforms",
  });

  // Bit-reversal pipeline
  const bitrevModule = device.createShaderModule({
    label: (label ?? "fft1d") + "_bitrevModule",
    code: BITREV_WGSL,
  });

  const bitrevBGL = device.createBindGroupLayout({
    label: (label ?? "fft1d") + "_bitrevBGL",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });

  const bitrevPipelineLayout = device.createPipelineLayout({
    label: (label ?? "fft1d") + "_bitrevPL",
    bindGroupLayouts: [bitrevBGL],
  });

  const bitrevPipeline = device.createComputePipeline({
    label: (label ?? "fft1d") + "_bitrevPipeline",
    layout: bitrevPipelineLayout,
    compute: {
      module: bitrevModule,
      entryPoint: "main",
    },
  });

  // FFT stage pipeline
  const fftModule = device.createShaderModule({
    label: (label ?? "fft1d") + "_fftModule",
    code: FFT_STAGE_WGSL,
  });

  const fftBGL = device.createBindGroupLayout({
    label: (label ?? "fft1d") + "_fftBGL",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });

  const fftPipelineLayout = device.createPipelineLayout({
    label: (label ?? "fft1d") + "_fftPL",
    bindGroupLayouts: [fftBGL],
  });

  const fftStagePipeline = device.createComputePipeline({
    label: (label ?? "fft1d") + "_fftStagePipeline",
    layout: fftPipelineLayout,
    compute: {
      module: fftModule,
      entryPoint: "main",
    },
  });

  // Scale & inverse back to real
  const scaleModule = device.createShaderModule({
    label: (label ?? "fft1d") + "_scaleModule",
    code: SCALE_INV_WGSL,
  });

  const scaleBGL = device.createBindGroupLayout({
    label: (label ?? "fft1d") + "_scaleBGL",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });

  const scalePipelineLayout = device.createPipelineLayout({
    label: (label ?? "fft1d") + "_scalePL",
    bindGroupLayouts: [scaleBGL],
  });

  const scalePipeline = device.createComputePipeline({
    label: (label ?? "fft1d") + "_scalePipeline",
    layout: scalePipelineLayout,
    compute: {
      module: scaleModule,
      entryPoint: "main",
    },
  });

  // Fill uniforms that don’t change across calls
  {
    const batchStrideReal = n;
    const batchStrideComplex = n;

    const bitrevUniformArray = new Uint32Array([
      n,
      logN,
      batchStrideReal,
      batchStrideComplex,
    ]);
    device.queue.writeBuffer(bitrevUniformBuf, 0, bitrevUniformArray.buffer);

    const scaleUniformArray = new Uint32Array([
      n,
      batchStrideComplex,
      batchStrideReal,
    ]);
    device.queue.writeBuffer(scaleUniformBuf, 0, scaleUniformArray.buffer);
  }

  return {
    device,
    n,
    batches,
    logN,
    complexBuf,
    bitrevUniformBuf,
    fftUniformBuf,
    scaleUniformBuf,
    bitrevPipeline,
    fftStagePipeline,
    scalePipeline,
    bitrevBGL,
    fftBGL,
    scaleBGL,
  };
}

// --- Public API: forward & inverse ---------------------------------------
//
// forwardFFT1DReal:
//   - takes real input buffer of size n * batches
//   - fills plan.complexBuf with the complex FFT (in natural order)
//
// inverseFFT1DReal:
//   - assumes plan.complexBuf currently holds spectral data
//   - performs inverse FFT and writes real part (with 1/n scaling)
//     back into dstReal.

export function forwardFFT1DReal(
  plan: FFT1DRealPlan,
  pass: GPUComputePassEncoder,
  srcReal: GPUBuffer,
) {
  const { device, n, batches, logN, complexBuf } = plan;

  // 1) Bit-reversed load: real -> complexBuf
  const bitrevBindGroup = device.createBindGroup({
    layout: plan.bitrevBGL,
    entries: [
      { binding: 0, resource: { buffer: srcReal } },
      { binding: 1, resource: { buffer: complexBuf } },
      { binding: 2, resource: { buffer: plan.bitrevUniformBuf } },
    ],
  });

  pass.setPipeline(plan.bitrevPipeline);
  pass.setBindGroup(0, bitrevBindGroup);

  {
    const numGroupsX = Math.ceil(n / WG_SIZE_X);
    pass.dispatchWorkgroups(numGroupsX, batches, 1);
  }

  // 2) DIT stages, in-place on complexBuf
  const numButterflies = n / 2;
  const numGroupsX = Math.ceil(numButterflies / WG_SIZE_X);

  const fftBindGroup = device.createBindGroup({
    layout: plan.fftBGL,
    entries: [
      { binding: 0, resource: { buffer: complexBuf } },
      { binding: 1, resource: { buffer: plan.fftUniformBuf } },
    ],
  });

  pass.setPipeline(plan.fftStagePipeline);
  pass.setBindGroup(0, fftBindGroup);

  for (let stage = 1; stage <= logN; stage++) {
    const fftUniformArray = new Uint32Array([
      n,
      stage,
      1,        // direction = +1 forward
      n,        // batchStride (complex elements per batch)
      0,        // _pad
    ]);
    device.queue.writeBuffer(plan.fftUniformBuf, 0, fftUniformArray.buffer);

    pass.dispatchWorkgroups(numGroupsX, batches, 1);
  }
}

export function inverseFFT1DReal(
  plan: FFT1DRealPlan,
  pass: GPUComputePassEncoder,
  dstReal: GPUBuffer,
) {
  const { device, n, batches, logN, complexBuf } = plan;

  // 1) DIT stages with direction = -1 (inverse)
  const numButterflies = n / 2;
  const numGroupsX = Math.ceil(numButterflies / WG_SIZE_X);

  const fftBindGroup = device.createBindGroup({
    layout: plan.fftBGL,
    entries: [
      { binding: 0, resource: { buffer: complexBuf } },
      { binding: 1, resource: { buffer: plan.fftUniformBuf } },
    ],
  });

  pass.setPipeline(plan.fftStagePipeline);
  pass.setBindGroup(0, fftBindGroup);

  for (let stage = 1; stage <= logN; stage++) {
    const fftUniformArray = new Uint32Array([
      n,
      stage,
      0xFFFFFFFF, // -1 as i32
      n,          // batchStride
      0,          // _pad
    ]);
    device.queue.writeBuffer(plan.fftUniformBuf, 0, fftUniformArray.buffer);

    pass.dispatchWorkgroups(numGroupsX, batches, 1);
  }

  // 2) Scale by 1/n and write real part to dstReal
  const scaleBindGroup = device.createBindGroup({
    layout: plan.scaleBGL,
    entries: [
      { binding: 0, resource: { buffer: complexBuf } },
      { binding: 1, resource: { buffer: dstReal } },
      { binding: 2, resource: { buffer: plan.scaleUniformBuf } },
    ],
  });

  pass.setPipeline(plan.scalePipeline);
  pass.setBindGroup(0, scaleBindGroup);

  {
    const numGroupsXReal = Math.ceil(n / WG_SIZE_X);
    pass.dispatchWorkgroups(numGroupsXReal, batches, 1);
  }
}
