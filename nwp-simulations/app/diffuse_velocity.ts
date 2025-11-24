// diffuse_velocity.ts
import type { SimDims, Params } from "./init_boussinesq";

export type DiffuseVelIO = {
  u: GPUBuffer;
  v: GPUBuffer;
  w: GPUBuffer;
  rhs_u: GPUBuffer;
  rhs_v: GPUBuffer;
  rhs_w: GPUBuffer;
};

export type DiffuseVelArtifacts = {
  dispatch: (pass: GPUComputePassEncoder, io: DiffuseVelIO) => void;
  resources: { pipeline: GPUComputePipeline; uniforms: GPUBuffer; workgroups: number };
};

export function makeDiffuseVelocity(opts: {
  device: GPUDevice;
  dims: SimDims;
  params: Params; // uses nu, dx, dy, dz
}): DiffuseVelArtifacts {
  const { device, dims, params } = opts;
  const { nx, ny, nz } = dims;
  const N = nx * ny * nz;
  const nu = (params.nu ?? 0.0);

  const wgsl = /* wgsl */`
// --- diffuse_velocity.wgsl ---
struct U {
  nx: u32, ny: u32, nz: u32, N: u32,
  sx: u32, sy: u32, sz: u32, _padU: u32,
  inv_dx2: f32, inv_dy2: f32, inv_dz2: f32, nu: f32,
};

@group(0) @binding(0) var<storage, read>       u_in  : array<f32>;
@group(0) @binding(1) var<storage, read>       v_in  : array<f32>;
@group(0) @binding(2) var<storage, read>       w_in  : array<f32>;
@group(0) @binding(3) var<storage, read_write> rhs_u : array<f32>;
@group(0) @binding(4) var<storage, read_write> rhs_v : array<f32>;
@group(0) @binding(5) var<storage, read_write> rhs_w : array<f32>;
@group(0) @binding(6) var<uniform>             Ubuf  : U;

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
  // let izp = wrap_inc(iz, Ubuf.nz);
  // let izm = wrap_dec(iz, Ubuf.nz);
  let izp = min(iz + 1u, Ubuf.nz - 1u);
let izm = max(iz - 1u, 0u);


  let i    = idx;
  let i_xp = iz * sz + iy * sy + ixp * sx;
  let i_xm = iz * sz + iy * sy + ixm * sx;
  let i_yp = iz * sz + iyp * sy + ix  * sx;
  let i_ym = iz * sz + iym * sy + ix  * sx;
  let i_zp = izp * sz + iy * sy + ix  * sx;
  let i_zm = izm * sz + iy * sy + ix  * sx;

  // Laplacian for u
  let u_c  = u_in[i];
  let u_xp = u_in[i_xp]; let u_xm = u_in[i_xm];
  let u_yp = u_in[i_yp]; let u_ym = u_in[i_ym];
  // let u_zp = u_in[i_zp]; let u_zm = u_in[i_zm];
    let u_zp_raw = u_in[i_zp];
  let u_zm_raw = u_in[i_zm];
  var u_zp = u_zp_raw;
  var u_zm = u_zm_raw;

  if (iz == 0u)          { u_zm = u_zp_raw; }
  if (iz == Ubuf.nz-1u)  { u_zp = u_zm_raw; }

  let Lu = (u_xp - 2.0*u_c + u_xm) * Ubuf.inv_dx2 +
           (u_yp - 2.0*u_c + u_ym) * Ubuf.inv_dy2 +
           (u_zp - 2.0*u_c + u_zm) * Ubuf.inv_dz2;

  // Laplacian for v
  let v_c  = v_in[i];
  let v_xp = v_in[i_xp]; let v_xm = v_in[i_xm];
  let v_yp = v_in[i_yp]; let v_ym = v_in[i_ym];
  // let v_zp = v_in[i_zp]; let v_zm = v_in[i_zm];
    let v_zp_raw = v_in[i_zp];
  let v_zm_raw = v_in[i_zm];
  var v_zp = v_zp_raw;
  var v_zm = v_zm_raw;

  if (iz == 0u)          { v_zm = v_zp_raw; }
  if (iz == Ubuf.nz-1u)  { v_zp = v_zm_raw; }

  let Lv = (v_xp - 2.0*v_c + v_xm) * Ubuf.inv_dx2 +
           (v_yp - 2.0*v_c + v_ym) * Ubuf.inv_dy2 +
           (v_zp - 2.0*v_c + v_zm) * Ubuf.inv_dz2;

  // Laplacian for w
  let w_c  = w_in[i];
  let w_xp = w_in[i_xp]; let w_xm = w_in[i_xm];
  let w_yp = w_in[i_yp]; let w_ym = w_in[i_ym];
  // let w_zp = w_in[i_zp]; let w_zm = w_in[i_zm];
    let w_zp_raw = w_in[i_zp];
  let w_zm_raw = w_in[i_zm];
  var w_zp = w_zp_raw;
  var w_zm = w_zm_raw;

  if (iz == 0u) {
    // w_-1 = -w_+1
    w_zm = -w_zp_raw;
  }
  if (iz == Ubuf.nz - 1u) {
    // w_+1 = -w_-1
    w_zp = -w_zm_raw;
  }

  let Lw = (w_xp - 2.0*w_c + w_xm) * Ubuf.inv_dx2 +
           (w_yp - 2.0*w_c + w_ym) * Ubuf.inv_dy2 +
           (w_zp - 2.0*w_c + w_zm) * Ubuf.inv_dz2;

  if (Ubuf.nu != 0.0) {
    rhs_u[i] = rhs_u[i] + Ubuf.nu * Lu;
    rhs_v[i] = rhs_v[i] + Ubuf.nu * Lv;
    rhs_w[i] = rhs_w[i] + Ubuf.nu * Lw;
  }
}
`;

  const module = device.createShaderModule({ code: wgsl, label: "diffuse_velocity_module" });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label: "diffuse_velocity_pipeline",
  });

  const uniformsAB = new ArrayBuffer(48); // 4*u32 + 4*u32 + 4*f32 = 48
  const u32 = new Uint32Array(uniformsAB);
  const f32 = new Float32Array(uniformsAB);
  u32[0] = nx >>> 0;  u32[1] = ny >>> 0;  u32[2] = nz >>> 0;  u32[3] = N >>> 0;
  u32[4] = 1;         u32[5] = nx >>> 0;  u32[6] = (nx*ny) >>> 0; u32[7] = 0;
  f32[8]  = 1.0 / (dims.dx * dims.dx);
  f32[9]  = 1.0 / (dims.dy * dims.dy);
  f32[10] = 1.0 / (dims.dz * dims.dz);
  f32[11] = nu;

  const uniforms = device.createBuffer({ size: uniformsAB.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "diffuse_velocity_uniforms" });
  device.queue.writeBuffer(uniforms, 0, uniformsAB);

  const workgroups = Math.ceil(N / 256);

  function dispatch(pass: GPUComputePassEncoder, io: DiffuseVelIO) {
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: io.u } },
        { binding: 1, resource: { buffer: io.v } },
        { binding: 2, resource: { buffer: io.w } },
        { binding: 3, resource: { buffer: io.rhs_u } },
        { binding: 4, resource: { buffer: io.rhs_v } },
        { binding: 5, resource: { buffer: io.rhs_w } },
        { binding: 6, resource: { buffer: uniforms } },
      ],
      label: "diffuse_velocity_bg",
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups);
  }

  return { dispatch, resources: { pipeline, uniforms, workgroups } };
}
