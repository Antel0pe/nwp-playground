// microphysics_saturation.ts
import type { SimDims } from "./init_boussinesq";

export type MicrophysicsArtifacts = {
  microphysics: (
    pass: GPUComputePassEncoder,
    theta_p: GPUBuffer,
    qv: GPUBuffer,
    qc: GPUBuffer,
    theta0: GPUBuffer,
    p0: GPUBuffer
  ) => void;
  resources: {
    pipeline: GPUComputePipeline;
    uniforms: GPUBuffer;
    workgroups: number;
  };
};

export function makeMicrophysicsSaturationAdjust(opts: {
  device: GPUDevice;
  dims: SimDims;
}): MicrophysicsArtifacts {
  const { device, dims } = opts;
  const { nx, ny, nz } = dims;
  const N = nx * ny * nz;

  const module = device.createShaderModule({
    label: "microphysics_saturation_module",
    code: /* wgsl */`
struct U {
  nx: u32, ny: u32, nz: u32, N: u32,
  sx: u32, sy: u32, sz: u32, _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> theta_p : array<f32>;
@group(0) @binding(1) var<storage, read_write> qv      : array<f32>;
@group(0) @binding(2) var<storage, read_write> qc      : array<f32>;
@group(0) @binding(3) var<storage, read>       theta0  : array<f32>;
@group(0) @binding(4) var<storage, read>       p0      : array<f32>;
@group(0) @binding(5) var<uniform>             Ubuf    : U;

// ---- constants mirroring Python version ----
const R_air      : f32 = 287.0;
const cp_air     : f32 = 1004.0;
const p_ref      : f32 = 1.0e5;
const Lv         : f32 = 2.5e6;
const eps        : f32 = 0.622;
const TOL        : f32 = 1e-12;
const QC_CRIT    : f32 = 1e-4;
const RAIN_FRAC  : f32 = 0.3;
const DT_NEWTON  : f32 = 0.05;
const MAX_ITER   : u32 = 6u;

fn exner(p: f32) -> f32 {
  // Π = (p / p_ref)^(R/cp)
  return pow(p / p_ref, R_air / cp_air);
}

fn saturation_mixing_ratio(T: f32, p: f32) -> f32 {
  // Bolton/Magnus over liquid water, T in K, p in Pa
  let Tc = T - 273.15;
  let es_hPa = 6.112 * exp(17.67 * Tc / (Tc + 243.5));
  let es = es_hPa * 100.0; // Pa
  return eps * es / (p - es);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= Ubuf.N) { return; }

  let nx = Ubuf.nx;
  let ny = Ubuf.ny;
  let sx = Ubuf.sx;
  let sy = Ubuf.sy;
  let sz = Ubuf.sz;

  // 3D indices (mainly for stride-based indexing)
  let ix = idx % nx;
  let iy = (idx / nx) % ny;
  let iz = idx / (nx * ny);

  let i = iz * sz + iy * sy + ix * sx;

  // --- local base state ---
  let p0_ijk     = p0[i];
  let Pi_ijk     = exner(p0_ijk);
  let c0_ijk     = Lv / (cp_air * Pi_ijk);

  let theta_pert = theta_p[i];
  let theta_base = theta0[i];
  let theta_ijk  = theta_base + theta_pert;

  var qv_ijk = qv[i];
  var qc_ijk = qc[i];

  let qt_i = qv_ijk + qc_ijk;
  let theta_l_i = theta_ijk - c0_ijk * qc_ijk;

  // Try unsaturated (qc_f = 0)
  let theta_unsat = theta_l_i;
  let T_unsat     = theta_unsat * Pi_ijk;
  let qs_unsat    = saturation_mixing_ratio(T_unsat, p0_ijk);

  var theta_f: f32 = 0.0;
  var qv_f   : f32 = 0.0;
  var qc_f   : f32 = 0.0;

  if (qt_i <= qs_unsat + TOL) {
    // Fully unsaturated
    qc_f    = 0.0;
    qv_f    = qt_i;
    theta_f = theta_unsat;
  } else {
    // Saturated: Newton iteration on qc_f
    qc_f = qt_i - qs_unsat;
    if (qc_f < 0.0) { qc_f = 0.0; }
    if (qc_f > qt_i) { qc_f = qt_i; }

    var it: u32 = 0u;
    loop {
      if (it >= MAX_ITER) { break; }

      theta_f = theta_l_i + c0_ijk * qc_f;
      let T_f  = theta_f * Pi_ijk;
      let qs_f = saturation_mixing_ratio(T_f, p0_ijk);

      let f = qt_i - qc_f - qs_f;

      // relative tolerance on |f|
      var qt_scale = qt_i;
      if (qt_scale <= 1e-12) {
        qt_scale = 1e-12;
      }
      let tol_scale = TOL * qt_scale;

      if (abs(f) < tol_scale) {
        break;
      }

      let qs_p   = saturation_mixing_ratio(T_f + DT_NEWTON, p0_ijk);
      let dqs_dT = (qs_p - qs_f) / DT_NEWTON;

      let df_dqc = -1.0 - dqs_dT * (c0_ijk * Pi_ijk);
      if (df_dqc == 0.0) {
        break;
      }

      var qc_newton = qc_f - f / df_dqc;

      if (qc_newton < 0.0) { qc_newton = 0.0; }
      if (qc_newton > qt_i) { qc_newton = qt_i; }

      qc_f = qc_newton;
      it = it + 1u;
    }

    theta_f = theta_l_i + c0_ijk * qc_f;
    qv_f    = qt_i - qc_f;
  }

  // --- autoconversion + rain-out ---
  if (qc_f > QC_CRIT) {
    let excess = qc_f - QC_CRIT;
    let rain   = RAIN_FRAC * excess;
    qc_f = qc_f - rain;
  }

  if (qv_f < 0.0) { qv_f = 0.0; }
  if (qc_f < 0.0) { qc_f = 0.0; }

  // --- write back: in-place update of theta_p, qv, qc ---
  theta_p[i] = theta_f - theta_base;
  qv[i]      = qv_f;
  qc[i]      = qc_f;
}
`,
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label: "microphysics_saturation_pipeline",
  });

  // Uniforms: dims & strides
  // struct U {
  //   nx, ny, nz, N : u32
  //   sx, sy, sz, _pad : u32
  // }
  const ab = new ArrayBuffer(32);
  const u32 = new Uint32Array(ab);
  u32[0] = nx >>> 0;
  u32[1] = ny >>> 0;
  u32[2] = nz >>> 0;
  u32[3] = N >>> 0;
  u32[4] = 1;                 // sx
  u32[5] = nx >>> 0;          // sy
  u32[6] = (nx * ny) >>> 0;   // sz
  u32[7] = 0;                 // _pad

  const uniforms = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "microphysics_saturation_uniforms",
  });
  device.queue.writeBuffer(uniforms, 0, ab);

  const workgroups = Math.ceil(N / 256);

  function microphysics(
    pass: GPUComputePassEncoder,
    theta_p: GPUBuffer,
    qv: GPUBuffer,
    qc: GPUBuffer,
    theta0: GPUBuffer,
    p0: GPUBuffer
  ) {
    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: theta_p } },
        { binding: 1, resource: { buffer: qv } },
        { binding: 2, resource: { buffer: qc } },
        { binding: 3, resource: { buffer: theta0 } },
        { binding: 4, resource: { buffer: p0 } },
        { binding: 5, resource: { buffer: uniforms } },
      ],
      label: "microphysics_saturation_bg",
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups);
  }

  return { microphysics, resources: { pipeline, uniforms, workgroups } };
}
