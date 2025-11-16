// project_velocity.ts
import type { SimDims } from "../init_boussinesq";
import { makeDivergence } from "./divergence";
import { makeJacobiPoisson } from "./jacobi_poisson";
import { makeGradSubtract } from "./grad_subtract";

export function makeProjection(opts:{ device: GPUDevice; dims: SimDims }) {
  const { device, dims } = opts;
  const N = dims.nx * dims.ny * dims.nz;
  const bytes = N * 4;

  // scratch: div, psi ping-pong
  const div = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  // psiA, psiB: start as zero (Jacobi from 0)
  const psiA = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  {
    const view = new Float32Array(psiA.getMappedRange());
    view.fill(0);
    psiA.unmap();
  }

  const psiB = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  {
    const view = new Float32Array(psiB.getMappedRange());
    view.fill(0);
    psiB.unmap();
  }

  const divergence = makeDivergence({ device, dims });
  const jacobi = makeJacobiPoisson({ device, dims });
  const gradSub = makeGradSubtract({ device, dims });

  function project(pass: GPUComputePassEncoder, u: GPUBuffer, v: GPUBuffer, w: GPUBuffer, iters = 40) {
    // div = ∂u+∂v+∂w
    divergence.dispatch(pass, { u, v, w, out_div: div });

    // initialize psi = 0 (one clear)
    // You already have a "CLEAR" utility. Bind it here once for psiA.
    // Or just run a quick copy from a zero buffer you keep around.
    // Minimal version: reuse your CLEAR pipeline outside this module.

    // Jacobi iterations (ping-pong psiA <-> psiB)
    var ping = psiA, pong = psiB;
    for (var k=0; k<iters; k++) {
      jacobi.dispatch(pass, { rhs_div: div, psi_in: ping, psi_out: pong });
      let t = ping; ping = pong; pong = t;
    }
    // ping holds the latest solution

    // velocity correction
    gradSub.dispatch(pass, { psi: ping, u, v, w });
  }

  return { project, resources: { div, psiA, psiB } };
}
