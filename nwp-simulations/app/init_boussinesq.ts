// init_boussinesq_webgpu.ts
// TypeScript CPU-side initialization that mirrors your Python setup
// It builds Float32Arrays for all 3D fields and uploads them to WebGPU buffers.
// Grid: nx × ny × nz (x,y,z) with (i + 0.5)*dx centering, same as your code.

export type SimDims = { nx: number; ny: number; nz: number; dx: number; dy: number; dz: number };
export type FieldBuffers = {
  // prognostic/state
  u: GPUBuffer; v: GPUBuffer; w: GPUBuffer;
  theta_p: GPUBuffer; qv: GPUBuffer; qc: GPUBuffer; pi: GPUBuffer;
  // diagnostic/background
  theta0: GPUBuffer; p0: GPUBuffer; qv_bg: GPUBuffer;
  dtheta0_dz: GPUBuffer;                 

  // rhs accumulators (tendencies) — cleared each step before recompute
  rhs_w: GPUBuffer;                      
  rhs_theta_p: GPUBuffer;               
  // surface targets (only first kmax=Nbl levels are meaningful, rest 0)
  theta_surf_target: GPUBuffer; qv_surf_target: GPUBuffer;
  bg_thermo: GPUBuffer;
  rhs_qv: GPUBuffer;
rhs_qc: GPUBuffer;
rhs_u: GPUBuffer;
rhs_v: GPUBuffer;
rho0: GPUBuffer;
inv_rho0: GPUBuffer;
};

export type Params = {
  Lv: number; cp: number; R: number; p_ref: number; eps: number; g: number;
  tau_damp_w: number; tau_rad: number; Nbl: number; tau_surf: number;
  qc_crit: number; rain_frac: number;
  dx: number; dy: number; dz: number;
  nu: number; kappa: number; Dq: number;
};

function exner(p: number, p_ref = 1.0e5, R = 287.0, cp = 1004.0) {
  return Math.pow(p / p_ref, R / cp);
}

// Saturation mixing ratio over liquid water (Pa-based Magnus/ Bolton-like)
function saturationMixingRatio(T: number, p: number, eps = 0.622) {
  // es in Pa (Magnus-Tetens form). Your Python used slightly different constants; this is close.
  const Tc = T - 273.15;
  const es_hPa = 6.112 * Math.exp((17.67 * Tc) / (Tc + 243.5));
  const es = es_hPa * 100.0;
  return eps * es / Math.max(1e-3, (p - es));
}

function idx(i: number, j: number, k: number, nx: number, ny: number) {
  return i + nx * (j + ny * k);
}

export async function initBoussinesq3D(device: GPUDevice) {
  // ------------------ Domain & grid ------------------
  const nx = 64, ny = 64, nz = 40;
  const Lx = 6400.0, Ly = 6400.0, Lz = 4000.0;
  const dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

  // Arrays (CPU)
  const N = nx * ny * nz;
  const u = new Float32Array(N);
  const v = new Float32Array(N);
  const w = new Float32Array(N);
  const theta_p = new Float32Array(N);
  const qv = new Float32Array(N);
  const qc = new Float32Array(N);
  const pi = new Float32Array(N);

  const theta0 = new Float32Array(N);
  const p0 = new Float32Array(N);
  const qv_bg = new Float32Array(N);

  // Surface target fields over the *whole* domain for simplicity (we'll only fill k < Nbl)
  const theta_surf_target = new Float32Array(N);
  const qv_surf_target = new Float32Array(N);

  const dtheta0_dz_arr = new Float32Array(N);   // per-cell gradient field
  const rhs_w_arr = new Float32Array(N);        // zeroed tendencies
  const rhs_theta_p_arr = new Float32Array(N);  // zeroed tendencies

  // Make sure you created dtheta0_dz_arr earlier (Float32Array(N).fill(0.003))
const thermoPacked = new Float32Array(3 * N);

const rhs_qv_arr = new Float32Array(N);
const rhs_qc_arr = new Float32Array(N);

const rhs_u_arr = new Float32Array(N);
const rhs_v_arr = new Float32Array(N);

const rho0 = new Float32Array(N);
const inv_rho0 = new Float32Array(N);

const nu = 10.0; // momentum
const kappa = 10.0; // thermal
const Dq = 10.0; // moisture


  // ------------------ Constants (match your Python) ------------------
  const theta0_surface = 300.0;             // K
  const dtheta0_dz = 0.003;                 // K/m (stable stratification)

  const R = 287.0;                          // J/(kg·K)
  const g = 9.81;                           // m/s^2
  const T_ref = 300.0;                      // K
  const H = (R * T_ref) / g;                // scale height ~ 8780 m

  const p_surf = 1.0e5;                     // Pa
  const cp = 1004.0;                        // J/(kg·K)
  const p_ref = 1.0e5;                      // Pa

  const RH_bg = 0.75;                        // background RH

  // ------------------ Stratification + humidity knobs ------------------

// Height of lid / inversion (m)
const z_lid = 2000.0;

// Potential temperature gradients (K/m)
// Below lid: weak/moderate stability (free convective layer)
const dtheta_dz_free = 0.001;   // 3 K/km, similar to what you had

// Above lid: stronger stability (acts as ceiling)
const dtheta_dz_lid  = 0.008;   // 8 K/km (tune this if you want a stronger/weaker lid)

// Background relative humidity profile
// Below lid: moist environment so edges aren't instantly shredded
const RH_free = 0.7            // 80% RH below lid

// At top of domain: much drier environment
const RH_top  = 0.2;            // 40% RH near top (can go lower if you want)

// Piecewise θ0(z): weakly stable below lid, strongly stable above
function theta0_profile(z: number): { theta0: number; dtheta_dz: number } {
  if (z <= z_lid) {
    const theta0 = theta0_surface + dtheta_dz_free * z;
    return { theta0, dtheta_dz: dtheta_dz_free };
  } else {
    const theta_lid = theta0_surface + dtheta_dz_free * z_lid;
    const theta0 = theta_lid + dtheta_dz_lid * (z - z_lid);
    return { theta0, dtheta_dz: dtheta_dz_lid };
  }
}

// Background RH(z): moist below lid, linearly drying to RH_top at the top
function RH_bg_profile(z: number, Lz: number): number {
  if (z <= z_lid) {
    return RH_free;
  } else {
    const alpha = Math.min(1.0, (z - z_lid) / Math.max(1.0, (Lz - z_lid)));
    // linear blend from RH_free at z_lid to RH_top at top
    return RH_free + (RH_top - RH_free) * alpha;
  }
}



  // Bubble (ellipsoidal)
  const xb = Lx * 0.5, yb = Ly * 0.5, zb = 300.0;
  const rb_xy = 600.0, rb_z = 300.0;

  const RH_bubble = 0.7;
  const theta_amp = 0.01;                    // K

  // Surface forcing params
  const Nbl = 2;                            // number of bottom levels forced
  const delta_theta_core = 0.1;             // K
  const RH_surf_core = 0.9;

  // Thermo/physics params you use elsewhere
  const Lv = 2.5e6, eps = 0.622;
  const tau_damp_w = 300.0;
  const tau_rad = 1800.0;
  const tau_surf = 1.0;
  const qc_crit = 1e-5;
  const rain_frac = 0.1;

  // ------------------ Precompute per-level z centers ------------------
  const zs = new Float32Array(nz);
  for (let k = 0; k < nz; k++) zs[k] = (k + 0.5) * dz;

  // for (let k = 0; k < nz; k++) {
  //   for (let j = 0; j < ny; j++) {
  //     for (let i = 0; i < nx; i++) {
  //       dtheta0_dz_arr[idx(i, j, k, nx, ny)] = dtheta0_dz; // 0.003
  //     }
  //   }
  // }

  // ------------------ Fill 3D fields ------------------
  // for (let k = 0; k < nz; k++) {
  //   const z = zs[k];
  //   const thetabase_k = theta0_surface + dtheta0_dz * z; // θ0(z)
  //   const pbase_k = p_surf * Math.exp(-z / H);           // p0(z)
  //   const Pi_k = exner(pbase_k, p_ref, R, cp);           // Π(z)
  //   const T0_k = thetabase_k * Pi_k;                     // T0(z)
  //   const qs_bg_k = saturationMixingRatio(T0_k, pbase_k, eps);
  //   const qv_bg_k = RH_bg * qs_bg_k;

  //   // -- y-level constants for surface forcing (k==0 specifics later) --

  //   for (let j = 0; j < ny; j++) {
  //     const y = (j + 0.5) * dy;
  //     for (let i = 0; i < nx; i++) {
  //       const x = (i + 0.5) * dx;
  //       const id = idx(i, j, k, nx, ny);

  //       // background fields
  //       theta0[id] = thetabase_k;
  //       p0[id] = pbase_k;
  //       qv_bg[id] = qv_bg_k;

  //       // bubble mask (ellipsoidal)
  //       const r2_xy = (x - xb) * (x - xb) + (y - yb) * (y - yb);
  //       const r2_z = (z - zb) * (z - zb);
  //       const bubble = Math.exp(-r2_xy / (2.0 * rb_xy * rb_xy) - r2_z / (2.0 * rb_z * rb_z));

  //       // initial anomalies
  //       theta_p[id] = theta_amp * bubble;              // warm anomaly

  //       // background saturation at this (z): use T0_k, pbase_k
  //       const qs_bg_here = qs_bg_k;                    // already per-k

  //       // bubble is closer to saturation
  //       const qv_bubble = qv_bg_k + (RH_bubble * qs_bg_here - qv_bg_k) * bubble;
  //       qv[id] = qv_bubble;

  //       // zero initial motion/cloud/pressure-perturb
  //       u[id] = 0.0; v[id] = 0.0; w[id] = 0.0; qc[id] = 0.0; pi[id] = 0.0;

  //       // Initialize surface targets only for k < Nbl, others = 0
  //       if (k < Nbl) {
  //         // Gaussian mask under bubble on the *lowest* level shape, extended to k < Nbl
  //         const r2_xy_surf = r2_xy; // same expression (z ignored)
  //         const rb_forcing = rb_xy;
  //         const forcing_mask = Math.exp(-r2_xy_surf / (2.0 * rb_forcing * rb_forcing));

  //         // theta target: background profile + warm anomaly under mask
  //         theta_surf_target[id] = thetabase_k + forcing_mask * delta_theta_core;

  //         // moisture blend to RH_surf_core using *surface* pressure and warmed surface θ
  //         const Pi_surf = exner(p_surf, p_ref, R, cp);
  //         const theta_core_xy = theta0_surface + delta_theta_core;     // scalar core θ
  //         const T_core_xy = theta_core_xy * Pi_surf;                    // warmed surface T
  //         const qs_core_xy = saturationMixingRatio(T_core_xy, p_surf, eps);

  //         // qv_bg at *surface* for blending baseline
  //         const qv_bg_surf = RH_bg * saturationMixingRatio(theta0_surface * Pi_surf, p_surf, eps);

  //         const qv_core_xy = RH_surf_core * qs_core_xy;
  //         qv_surf_target[id] = qv_bg_k + forcing_mask * (qv_core_xy - qv_bg_surf);
  //       } else {
  //         theta_surf_target[id] = 0.0;
  //         qv_surf_target[id] = 0.0;
  //       }
  //     }
  //   }
  // }
  for (let k = 0; k < nz; k++) {
  const z = zs[k];

  // Get θ0(z) and local dθ0/dz from our 2-layer profile
  const { theta0: thetabase_k, dtheta_dz } = theta0_profile(z);

  const pbase_k = p_surf * Math.exp(-z / H);      // p0(z)
  const Pi_k = exner(pbase_k, p_ref, R, cp);      // Π(z)
  const T0_k = thetabase_k * Pi_k;                // T0(z)

  // Background RH(z): moist below lid, drying aloft
  const RH_bg_k = RH_bg_profile(z, Lz);
  const qs_bg_k = saturationMixingRatio(T0_k, pbase_k, eps);
  const qv_bg_k = RH_bg_k * qs_bg_k;

  for (let j = 0; j < ny; j++) {
    const y = (j + 0.5) * dy;
    for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) * dx;
      const id = idx(i, j, k, nx, ny);

      // store background and local gradient
      theta0[id] = thetabase_k;
      p0[id] = pbase_k;
      qv_bg[id] = qv_bg_k;
      dtheta0_dz_arr[id] = dtheta_dz;

      // density
      rho0[id] = p0[id] / (R * T0_k);
      inv_rho0[id] = 1.0 / rho0[id];

      // --- bubble + ICs as before ---

      const r2_xy = (x - xb) * (x - xb) + (y - yb) * (y - yb);
      const r2_z  = (z - zb) * (z - zb);
      const bubble = Math.exp(-r2_xy / (2.0 * rb_xy * rb_xy) - r2_z / (2.0 * rb_z * rb_z));

      theta_p[id] = theta_amp * bubble;

      const qs_bg_here = qs_bg_k;
      const qv_bubble = qv_bg_k + (RH_bubble * qs_bg_here - qv_bg_k) * bubble;
      qv[id] = qv_bubble;

      u[id] = 0.0; v[id] = 0.0; w[id] = 0.0; qc[id] = 0.0; pi[id] = 0.0;

      // surface targets section unchanged (if you keep it disabled in the stepper, this is just data)
      if (k < Nbl) {
        const r2_xy_surf = r2_xy;
        const rb_forcing = rb_xy;
        const forcing_mask = Math.exp(-r2_xy_surf / (2.0 * rb_forcing * rb_forcing));

        theta_surf_target[id] = thetabase_k + forcing_mask * delta_theta_core;

        const Pi_surf = exner(p_surf, p_ref, R, cp);
        const theta_core_xy = theta0_surface + delta_theta_core;
        const T_core_xy = theta_core_xy * Pi_surf;
        const qs_core_xy = saturationMixingRatio(T_core_xy, p_surf, eps);

        const qv_bg_surf = RH_free * saturationMixingRatio(theta0_surface * Pi_surf, p_surf, eps);
        const qv_core_xy = RH_surf_core * qs_core_xy;
        qv_surf_target[id] = qv_bg_k + forcing_mask * (qv_core_xy - qv_bg_surf);
      } else {
        theta_surf_target[id] = 0.0;
        qv_surf_target[id] = 0.0;
      }
    }
  }
}


    thermoPacked.set(theta0, 0 * N);
thermoPacked.set(dtheta0_dz_arr, 1 * N);
thermoPacked.set(qv_bg, 2 * N);

  // ------------------ Upload helpers ------------------
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  function makeBuf(data: Float32Array) {
    const buf = device.createBuffer({ size: data.byteLength, usage });
    const ab = data.buffer as ArrayBuffer;         // ensure ArrayBuffer, not SharedArrayBuffer
     device.queue.writeBuffer(buf, 0, ab, data.byteOffset, data.byteLength);
    return buf;
  }

  // ------------------ Create GPU buffers ------------------
  const fields: FieldBuffers = {
    u: makeBuf(u), v: makeBuf(v), w: makeBuf(w),
    theta_p: makeBuf(theta_p), qv: makeBuf(qv), qc: makeBuf(qc), pi: makeBuf(pi),
    theta0: makeBuf(theta0), p0: makeBuf(p0), qv_bg: makeBuf(qv_bg),
    theta_surf_target: makeBuf(theta_surf_target), qv_surf_target: makeBuf(qv_surf_target),
  dtheta0_dz: makeBuf(dtheta0_dz_arr),   
  rhs_w: makeBuf(rhs_w_arr),           
  rhs_theta_p: makeBuf(rhs_theta_p_arr),   
  bg_thermo: makeBuf(thermoPacked),   
  rhs_qv: makeBuf(rhs_qv_arr),
rhs_qc: makeBuf(rhs_qc_arr),
rhs_u: makeBuf(rhs_u_arr),
rhs_v: makeBuf(rhs_v_arr),
rho0: makeBuf(rho0),
inv_rho0: makeBuf(inv_rho0),
  };

  const params: Params = { nu, kappa, Dq, dx, dy, dz, Lv, cp, R, p_ref, eps, g, tau_damp_w, tau_rad, Nbl, tau_surf, qc_crit, rain_frac };

  const dims: SimDims = { nx, ny, nz, dx, dy, dz };

  return { fields, params, dims };
}

// Example usage elsewhere:
// const adapter = await navigator.gpu.requestAdapter();
// const device = await adapter!.requestDevice();
// const { fields, params, dims } = await initBoussinesq3D(device);
// Now bind these buffers into your compute pipelines (advection, diffusion, microphysics, projection, etc).

