// advect_scalar.ts
import type { SimDims, Params } from "./init_boussinesq";

export type AdvectIO = {
  phi: GPUBuffer;      // scalar to advect (theta_p, qv, or qc)
  u: GPUBuffer;
  v: GPUBuffer;
  w: GPUBuffer;
  rhs_out: GPUBuffer;  // accumulate tendency into this buffer
};

export type AdvectArtifacts = {
  dispatch: (pass: GPUComputePassEncoder, io: AdvectIO) => void;
  resources: { pipeline: GPUComputePipeline; uniforms: GPUBuffer; workgroups: number };
};

export function makeAdvectScalar(opts: {
  device: GPUDevice;
  dims: SimDims;
  params: Params;   // uses dx,dy,dz
}): AdvectArtifacts {
  const { device, dims, params } = opts;
  const { nx, ny, nz } = dims;
  const N = nx * ny * nz;

  const module = device.createShaderModule({
    code: /* wgsl */`
// --- advect_scalar.wgsl ---
struct AdvUniforms {
  nx: u32, ny: u32, nz: u32, N: u32,
  sx: u32, sy: u32, sz: u32, _uPad: u32,
  inv_dx: f32, inv_dy: f32, inv_dz: f32, _fPad: f32,
};

@group(0) @binding(0) var<storage, read>       phi  : array<f32>;
@group(0) @binding(1) var<storage, read>       u    : array<f32>;
@group(0) @binding(2) var<storage, read>       v    : array<f32>;
@group(0) @binding(3) var<storage, read>       w    : array<f32>;
@group(0) @binding(4) var<storage, read_write> rhs  : array<f32>;
@group(0) @binding(5) var<uniform>             U    : AdvUniforms;

fn wrap_inc(i: u32, n: u32) -> u32 { return select(i + 1u, 0u, i + 1u >= n); }
fn wrap_dec(i: u32, n: u32) -> u32 { return select(i - 1u, n - 1u, i == 0u); }

@compute @workgroup_size(256)
fn advect_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= U.N) { return; }

  let ix = idx % U.nx;
  let iy = (idx / U.nx) % U.ny;
  let iz = idx / (U.nx * U.ny);

  let sx = U.sx;          // 1
  let sy = U.sy;          // nx
  let sz = U.sz;          // nx*ny

  let ixp = wrap_inc(ix, U.nx);
  let ixm = wrap_dec(ix, U.nx);
  let iyp = wrap_inc(iy, U.ny);
  let iym = wrap_dec(iy, U.ny);
  let izp = wrap_inc(iz, U.nz);
  let izm = wrap_dec(iz, U.nz);

  let i     = idx;
  let i_xp  = iz * sz + iy * sy + ixp * sx;
  let i_xm  = iz * sz + iy * sy + ixm * sx;
  let i_yp  = iz * sz + iyp * sy + ix * sx;
  let i_ym  = iz * sz + iym * sy + ix * sx;
  let i_zp  = izp * sz + iy * sy + ix * sx;
  let i_zm  = izm * sz + iy * sy + ix * sx;

  let phi_c  = phi[i];
  let phi_xp = phi[i_xp];
  let phi_xm = phi[i_xm];
  let phi_yp = phi[i_yp];
  let phi_ym = phi[i_ym];
  let phi_zp = phi[i_zp];
  let phi_zm = phi[i_zm];

  // face-centered velocities (averages)
  let ux_p = 0.5 * (u[i]    + u[i_xp]);
  let ux_m = 0.5 * (u[i_xm] + u[i]);
  let vy_p = 0.5 * (v[i]    + v[i_yp]);
  let vy_m = 0.5 * (v[i_ym] + v[i]);
  let wz_p = 0.5 * (w[i]    + w[i_zp]);
  let wz_m = 0.5 * (w[i_zm] + w[i]);

  // upwind picks
  let phi_x_up_p = select(phi_xp, phi_c,  ux_p > 0.0);
  let phi_x_up_m = select(phi_c,  phi_xm, ux_m > 0.0);
  let phi_y_up_p = select(phi_yp, phi_c,  vy_p > 0.0);
  let phi_y_up_m = select(phi_c,  phi_ym, vy_m > 0.0);
  let phi_z_up_p = select(phi_zp, phi_c,  wz_p > 0.0);
  let phi_z_up_m = select(phi_c,  phi_zm, wz_m > 0.0);

  // fluxes
  let Fx_p = ux_p * phi_x_up_p;
  let Fx_m = ux_m * phi_x_up_m;
  let Fy_p = vy_p * phi_y_up_p;
  let Fy_m = vy_m * phi_y_up_m;
  let Fz_p = wz_p * phi_z_up_p;
  let Fz_m = wz_m * phi_z_up_m;

  // divergence
  let dFx = (Fx_p - Fx_m) * U.inv_dx;
  let dFy = (Fy_p - Fy_m) * U.inv_dy;
  let dFz = (Fz_p - Fz_m) * U.inv_dz;

  rhs[i] = rhs[i] - (dFx + dFy + dFz);
}
`,
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "advect_scalar" },
  });

  // uniforms
  const uniformsAB = new ArrayBuffer(48);
  const u32 = new Uint32Array(uniformsAB);
  const f32 = new Float32Array(uniformsAB);
  u32[0] = nx >>> 0; u32[1] = ny >>> 0; u32[2] = nz >>> 0; u32[3] = N >>> 0;
  u32[4] = 1;        u32[5] = nx >>> 0; u32[6] = (nx*ny) >>> 0; u32[7] = 0;
  f32[8] = 1.0 / (params.dx as number);
  f32[9] = 1.0 / (params.dy as number);
  f32[10]= 1.0 / (params.dz as number);

  const uniforms = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniforms, 0, uniformsAB);

  const workgroups = Math.ceil(N / 256);

  function dispatch(pass: GPUComputePassEncoder, io: AdvectIO) {
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: io.phi } },
        { binding: 1, resource: { buffer: io.u } },
        { binding: 2, resource: { buffer: io.v } },
        { binding: 3, resource: { buffer: io.w } },
        { binding: 4, resource: { buffer: io.rhs_out } },
        { binding: 5, resource: { buffer: uniforms } },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups);
  }

  return { dispatch, resources: { pipeline, uniforms, workgroups } };
}
