// diffuse_scalar.ts
import type { SimDims } from "./init_boussinesq";

export type DiffuseScalarIO = {
  phi: GPUBuffer;      // input scalar field
  rhs_out: GPUBuffer;  // accumulate ν∇²φ here
};

export type DiffuseScalarArtifacts = {
  dispatch: (pass: GPUComputePassEncoder, io: DiffuseScalarIO) => void;
  resources: { pipeline: GPUComputePipeline; uniforms: GPUBuffer; workgroups: number };
};

/**
 * Build a scalar-diffusion kernel with a fixed coefficient (coeff).
 * Call it for theta' (kappa) and for moisture (Dq).
 */
export function makeDiffuseScalar(opts: {
  device: GPUDevice;
  dims: SimDims;
  coeff: number; // kappa or Dq
}): DiffuseScalarArtifacts {
  const { device, dims, coeff } = opts;
  const { nx, ny, nz, dx, dy, dz } = dims;
  const N = nx * ny * nz;

  const module = device.createShaderModule({
    label: "diffuse_scalar_module",
    code: /* wgsl */`
struct U {
  nx: u32, ny: u32, nz: u32, N: u32,
  sx: u32, sy: u32, sz: u32, _padU: u32,
  inv_dx2: f32, inv_dy2: f32, inv_dz2: f32, coeff: f32,
};

@group(0) @binding(0) var<storage, read>       phi   : array<f32>;
@group(0) @binding(1) var<storage, read_write> rhs   : array<f32>;
@group(0) @binding(2) var<uniform>             Ubuf  : U;

fn wrap_inc(i: u32, n: u32) -> u32 { return select(i + 1u, 0u, i + 1u >= n); }
fn wrap_dec(i: u32, n: u32) -> u32 { return select(i - 1u, n - 1u, i == 0u); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= Ubuf.N) { return; }

  let nx = Ubuf.nx; let ny = Ubuf.ny;
  let sx = Ubuf.sx; let sy = Ubuf.sy; let sz = Ubuf.sz;

  let ix = idx % nx;
  let iy = (idx / nx) % ny;
  let iz = idx / (nx * ny);

  let ixp = wrap_inc(ix, nx);
  let ixm = wrap_dec(ix, nx);
  let iyp = wrap_inc(iy, ny);
  let iym = wrap_dec(iy, ny);
  let izp = wrap_inc(iz, Ubuf.nz);
  let izm = wrap_dec(iz, Ubuf.nz);

  let i    = idx;
  let i_xp = iz * sz + iy * sy + ixp * sx;
  let i_xm = iz * sz + iy * sy + ixm * sx;
  let i_yp = iz * sz + iyp * sy + ix  * sx;
  let i_ym = iz * sz + iym * sy + ix  * sx;
  let i_zp = izp * sz + iy * sy + ix  * sx;
  let i_zm = izm * sz + iy * sy + ix  * sx;

  let c   = phi[i];
  let xp  = phi[i_xp]; let xm = phi[i_xm];
  let yp  = phi[i_yp]; let ym = phi[i_ym];
  let zp  = phi[i_zp]; let zm = phi[i_zm];

  let Lap = (xp - 2.0*c + xm) * Ubuf.inv_dx2 +
            (yp - 2.0*c + ym) * Ubuf.inv_dy2 +
            (zp - 2.0*c + zm) * Ubuf.inv_dz2;

  if (Ubuf.coeff != 0.0) {
    rhs[i] = rhs[i] + Ubuf.coeff * Lap;
  }
}
`,
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label: "diffuse_scalar_pipeline",
  });

  // uniforms
  const ab = new ArrayBuffer(48);
  const u32 = new Uint32Array(ab);
  const f32 = new Float32Array(ab);
  u32[0] = nx >>> 0; u32[1] = ny >>> 0; u32[2] = nz >>> 0; u32[3] = N >>> 0;
  u32[4] = 1;        u32[5] = nx >>> 0; u32[6] = (nx*ny) >>> 0; u32[7] = 0;
  f32[8]  = 1.0 / (dx * dx);
  f32[9]  = 1.0 / (dy * dy);
  f32[10] = 1.0 / (dz * dz);
  f32[11] = coeff;

  const uniforms = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "diffuse_scalar_uniforms" });
  device.queue.writeBuffer(uniforms, 0, ab);

  const workgroups = Math.ceil(N / 256);

  function dispatch(pass: GPUComputePassEncoder, io: DiffuseScalarIO) {
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: io.phi } },
        { binding: 1, resource: { buffer: io.rhs_out } },
        { binding: 2, resource: { buffer: uniforms } },
      ],
      label: "diffuse_scalar_bg",
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups);
  }

  return { dispatch, resources: { pipeline, uniforms, workgroups } };
}
