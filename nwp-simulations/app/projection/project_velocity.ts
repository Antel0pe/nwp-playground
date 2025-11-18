// project_velocity.ts
import type { SimDims } from "../init_boussinesq";
import { makeDivergence } from "./divergence";
import { makeJacobiPoisson } from "./jacobi_poisson";
import { makeGradSubtract } from "./grad_subtract";

const CLEAR_WGSL = /* wgsl */`
struct Uniforms { N: u32, _pad: vec3<u32>, };
@group(0) @binding(0) var<storage, read_write> buf : array<f32>;
@group(0) @binding(1) var<uniform>             U   : Uniforms;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }
  buf[i] = 0.0;
}
`;

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
  let psiA = device.createBuffer({
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

  // --- CLEAR pipeline + uniforms for psiA ----
  const clearModule = device.createShaderModule({ code: CLEAR_WGSL });
  const clearPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: clearModule, entryPoint: "main" },
  });

  const clearUniformsAB = new ArrayBuffer(32);
  {
    const u32 = new Uint32Array(clearUniformsAB);
    u32[0] = N >>> 0;
  }
  const clearUniforms = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(clearUniforms, 0, clearUniformsAB);

  const wg = Math.ceil(N / 256);

  const bgClearPsiA = device.createBindGroup({
    layout: clearPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: psiA } },
      { binding: 1, resource: { buffer: clearUniforms } },
    ],
  });

  function project(pass: GPUComputePassEncoder, u: GPUBuffer, v: GPUBuffer, w: GPUBuffer, iters = 40) {
    // div = ∂u+∂v+∂w
    divergence.dispatch(pass, { u, v, w, out_div: div });

        // 2) reset psiA to zero EVERY CALL (match Python psi = 0)
    pass.setPipeline(clearPipeline);
    pass.setBindGroup(0, bgClearPsiA);
    pass.dispatchWorkgroups(wg);

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
