// debug_divergence.ts
import type { SimDims } from "../init_boussinesq";
import { makeDivergence } from "./divergence";

export function makeDivergenceDebugger(opts: { device: GPUDevice; dims: SimDims }) {
  const { device, dims } = opts;
  const N = dims.nx * dims.ny * dims.nz;
  const bytes = N * 4;

  // Storage buffer to hold divergence on GPU
  const divBuf = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Readback buffer for CPU inspection
  const readBuf = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const divergence = makeDivergence({ device, dims });

  async function logDivergence(label: string, u: GPUBuffer, v: GPUBuffer, w: GPUBuffer) {
    // 1. Run divergence kernel into divBuf
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    divergence.dispatch(pass, { u, v, w, out_div: divBuf });
    pass.end();

    // 2. Copy to readBuf
    encoder.copyBufferToBuffer(divBuf, 0, readBuf, 0, bytes);
    device.queue.submit([encoder.finish()]);

    // 3. Map and compute simple stats
    await readBuf.mapAsync(GPUMapMode.READ);
    const arr = new Float32Array(readBuf.getMappedRange());

    let maxAbs = 0;
    let sum = 0;
    let sumSq = 0;

    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      const av = Math.abs(v);
      if (av > maxAbs) maxAbs = av;
      sum += v;
      sumSq += v * v;
    }

    const mean = sum / arr.length;
    const rms = Math.sqrt(sumSq / arr.length);

    console.log(
      `[divergence] ${label} | maxAbs=${maxAbs.toExponential(4)}, ` +
      `mean=${mean.toExponential(4)}, rms=${rms.toExponential(4)}`
    );

    readBuf.unmap();
  }

  return { logDivergence };
}
