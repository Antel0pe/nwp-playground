// fft_project_velocity.ts

import { createKspaceProjectionPipeline } from "./project_velocity_kspace";

export type FFT3D = {
  // forward: real-space scalar field -> complex spectral field
  forward: (
    encoder: GPUCommandEncoder,
    real: GPUBuffer,       // length nx*ny*nz, f32
    complexOut: GPUBuffer  // length nx*ny*nz, vec2<f32>
  ) => void;

  // inverse: complex spectral field -> real-space scalar field
  inverse: (
    encoder: GPUCommandEncoder,
    complexIn: GPUBuffer,  // length nx*ny*nz, vec2<f32>
    realOut: GPUBuffer     // length nx*ny*nz, f32
  ) => void;
};

export type FFTProjector = {
  // Mutates u, v, w in-place (real-space) to their projected versions.
  // Also fills psiReal with the scalar potential psi in real space.
  project: (
    encoder: GPUCommandEncoder,
    u: GPUBuffer,
    v: GPUBuffer,
    w: GPUBuffer,
    psiReal: GPUBuffer
  ) => void;
};

export function makeFFTProjector(opts: {
  device: GPUDevice;
  nx: number;
  ny: number;
  nz: number;
  dx: number;
  dy: number;
  dz: number;
  fft: FFT3D;  // you plug your FFT implementation here
}): FFTProjector {
  const { device, nx, ny, nz, dx, dy, dz, fft } = opts;
  const N = nx * ny * nz;

  // --- 1) Precompute kx, ky, kz (same as _wavenumbers + np.fft.fftfreq) ---

  function fftfreq(n: number, d: number): Float32Array {
    // Same convention as np.fft.fftfreq(n, d):
    // k = [0, 1, ..., n//2-1, -n//2, ..., -1] / (n*d)
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
    label: "k_x",
  });
  const kyBuf = device.createBuffer({
    size: ny * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "k_y",
  });
  const kzBuf = device.createBuffer({
    size: nz * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "k_z",
  });

  device.queue.writeBuffer(kxBuf, 0, kxArr as GPUAllowSharedBufferSource);
  device.queue.writeBuffer(kyBuf, 0, kyArr as GPUAllowSharedBufferSource);
  device.queue.writeBuffer(kzBuf, 0, kzArr as GPUAllowSharedBufferSource);

  // --- 2) Allocate spectral buffers for U, V, W, psi_k (complex) ---

  const complexSize = N * 2 * 4; // vec2<f32>
  const Uc = device.createBuffer({
    size: complexSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: "Uc_k",
  });
  const Vc = device.createBuffer({
    size: complexSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: "Vc_k",
  });
  const Wc = device.createBuffer({
    size: complexSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: "Wc_k",
  });
  const psi_k = device.createBuffer({
    size: complexSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: "psi_k",
  });

  // dims uniform
  const dimsBuf = device.createBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "kspace_dims",
  });
  {
    const tmp = new Uint32Array([nx, ny, nz, 0]);
    device.queue.writeBuffer(dimsBuf, 0, tmp);
  }

  // --- 3) K-space projection pipeline + bind group ---

  const { pipeline, bindGroupLayout } = createKspaceProjectionPipeline(device);

  function makeBindGroup(): GPUBindGroup {
    return device.createBindGroup({
      layout: bindGroupLayout,
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
  }

  const bindGroup = makeBindGroup();

  // --- 4) Public project() function ----------------------------------

  function project(
    encoder: GPUCommandEncoder,
    u: GPUBuffer,
    v: GPUBuffer,
    w: GPUBuffer,
    psiReal: GPUBuffer
  ) {
    // 1) Forward 3D FFT: u,v,w (real) -> Uc,Vc,Wc (complex)
    fft.forward(encoder, u, Uc);
    fft.forward(encoder, v, Vc);
    fft.forward(encoder, w, Wc);

    // 2) K-space projection kernel
    {
      const pass = encoder.beginComputePass({
        label: "project_velocity_kspace_pass",
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);

      const workgroupSize = 256;
      const numGroups = Math.ceil(N / workgroupSize);
      pass.dispatchWorkgroups(numGroups, 1, 1);
      pass.end();
    }

    // 3) Inverse 3D FFT: Uc,Vc,Wc,psi_k -> u,v,w,psiReal (real)
    fft.inverse(encoder, Uc, u);
    fft.inverse(encoder, Vc, v);
    fft.inverse(encoder, Wc, w);
    fft.inverse(encoder, psi_k, psiReal);
  }

  return { project };
}
