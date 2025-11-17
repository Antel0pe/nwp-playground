// project_velocity_kspace.ts

export function createKspaceProjectionPipeline(device: GPUDevice) {
  const module = device.createShaderModule({
    label: "project_velocity_kspace_module",
    code: /* wgsl */`
struct Dims {
  nx: u32,
  ny: u32,
  nz: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> U : array<vec2<f32>>;  // complex U(k)
@group(0) @binding(1) var<storage, read_write> V : array<vec2<f32>>;  // complex V(k)
@group(0) @binding(2) var<storage, read_write> W : array<vec2<f32>>;  // complex W(k)
@group(0) @binding(3) var<storage, read_write> PSI : array<vec2<f32>>; // complex psi_k
@group(0) @binding(4) var<storage, read> kx_arr : array<f32>;         // length nx
@group(0) @binding(5) var<storage, read> ky_arr : array<f32>;         // length ny
@group(0) @binding(6) var<storage, read> kz_arr : array<f32>;         // length nz
@group(0) @binding(7) var<uniform> dims : Dims;

fn linear_index(ix: u32, iy: u32, iz: u32, nx: u32, ny: u32, nz: u32) -> u32 {
  return iz * (ny * nx) + iy * nx + ix;
}

// complex helpers

fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x + b.x, a.y + b.y);
}

fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x - b.x, a.y - b.y);
}

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

fn c_mul_real(a: vec2<f32>, s: f32) -> vec2<f32> {
  return vec2<f32>(a.x * s, a.y * s);
}

// multiply by i (0 + i)
fn c_mul_i(a: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(-a.y, a.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nx = dims.nx;
  let ny = dims.ny;
  let nz = dims.nz;

  let N: u32 = nx * ny * nz;
  let idx: u32 = gid.x;
  if (idx >= N) {
    return;
  }

  // Recover (ix, iy, iz) from linear index
  let iz = idx / (ny * nx);
  let rem = idx - iz * (ny * nx);
  let iy = rem / nx;
  let ix = rem - iy * nx;

  let kx = kx_arr[ix];
  let ky = ky_arr[iy];
  let kz = kz_arr[iz];

  let k2 = kx * kx + ky * ky + kz * kz;

  let Uk = U[idx];
  let Vk = V[idx];
  let Wk = W[idx];

  // div_k = i*(kx*U + ky*V + kz*W)
  // scale complex fields by real k components and sum:
  let kxU = c_mul_real(Uk, kx);
  let kyV = c_mul_real(Vk, ky);
  let kzW = c_mul_real(Wk, kz);

  var div_k = c_add(kxU, c_add(kyV, kzW));
  div_k = c_mul_i(div_k);  // multiply by i

  var psi_k = vec2<f32>(0.0, 0.0);

  if (k2 != 0.0) {
    // psi_k = -div_k / k2   (division by real scalar)
    psi_k = c_mul_real(div_k, -1.0 / k2);
  } else {
    psi_k = vec2<f32>(0.0, 0.0); // zero-mean gauge
  }

  PSI[idx] = psi_k;

  // Gradient of psi in spectral space:
  // Gx_k = i * kx * psi_k
  // Gy_k = i * ky * psi_k
  // Gz_k = i * kz * psi_k
  let kx_psi = c_mul_real(psi_k, kx);
  let ky_psi = c_mul_real(psi_k, ky);
  let kz_psi = c_mul_real(psi_k, kz);

  let Gx_k = c_mul_i(kx_psi);
  let Gy_k = c_mul_i(ky_psi);
  let Gz_k = c_mul_i(kz_psi);

  // Corrected velocities: Uc = U - Gx_k, etc.
  U[idx] = c_sub(Uk, Gx_k);
  V[idx] = c_sub(Vk, Gy_k);
  W[idx] = c_sub(Wk, Gz_k);
}
`,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: "project_velocity_kspace_bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // U
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // V
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // W
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // PSI
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // kx
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // ky
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // kz
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // dims
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "project_velocity_kspace_pl",
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createComputePipeline({
    label: "project_velocity_kspace_pipeline",
    layout: pipelineLayout,
    compute: {
      module,
      entryPoint: "main",
    },
  });

  return { pipeline, bindGroupLayout };
}
