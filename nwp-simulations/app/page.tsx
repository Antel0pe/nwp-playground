"use client";

import { useEffect, useRef, useState } from "react";
import { Pane } from "tweakpane";
import { initBoussinesq3D } from "./init_boussinesq";
import { makeStepRK2 } from "./step_rk2";
import { makeComputeRhs } from "./compute_rhs";

// Assume fields.w is your GPUBuffer
async function debugPrintBufferMax(device: GPUDevice, buffer: GPUBuffer) {
  const readBuffer = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size);
  device.queue.submit([encoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const arr = new Float32Array(readBuffer.getMappedRange());

  const maxVal = Math.max(...arr);
  console.log("Max value in buffer:", maxVal);
  console.log("First few values:", arr.slice(0, 10));

  readBuffer.unmap();
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const paneRef = useRef<HTMLDivElement | null>(null);
  const [accumulatedTime, setAccumulatedTime] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!("gpu" in navigator)) {
      console.error("WebGPU not supported in this browser.");
      return;
    }

    const init = async () => {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.error("Failed to get GPU adapter.");
        return;
      }

      const device = await adapter.requestDevice();
      device.pushErrorScope("validation");
      device.lost.then((info) => {
        console.error("WebGPU device lost:", info.message, "reason:", info.reason);
      });
      device.onuncapturederror = (e) => {
        console.error("Uncaptured WebGPU error:", e.error?.message);
      };

      const context = canvas.getContext("webgpu");
      if (!context) {
        console.error("Failed to get WebGPU context.");
        return;
      }

      const { fields, params, dims } = await initBoussinesq3D(device);
      const { nx, ny, nz } = dims;

      // ------------------------------------------------------------------
      // OUTPUT TEXTURE (nx x nz) FOR DISPLAY ONLY
      // ------------------------------------------------------------------
      const outTex = device.createTexture({
        size: { width: nx, height: nz },
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      });
      const outView = outTex.createView();

      const computeRhs = makeComputeRhs({ device, fields: fields as any, dims, params });

      const rk2 = makeStepRK2({
        device,
        fields,
        dims,
        params,
        computeRhs,
      });

      const dt = 0.5;

      // ------------------------------------------------------------------
      // COMPUTE PIPELINE: theta slice to outTex
      // ------------------------------------------------------------------
      const thetaModule = device.createShaderModule({
        code: /* wgsl */ `
struct Dims { data: array<u32> }; // [0]=nx, [1]=ny, [2]=nz, [3]=j

struct ViewUniforms {
  mode: u32,
  showClouds: u32,  // 0 = off, 1 = on
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> theta_p : array<f32>;
@group(0) @binding(1) var outImg : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> dims : Dims;
@group(0) @binding(3) var<storage, read> qc : array<f32>;
@group(0) @binding(4) var<storage, read> qv : array<f32>;       // water vapour
@group(0) @binding(5) var<storage, read> theta0 : array<f32>;   // base theta
@group(0) @binding(6) var<storage, read> p0 : array<f32>;       // base p
@group(0) @binding(7) var<uniform>      V : ViewUniforms;

fn NX() -> u32 { return dims.data[0]; }
fn NY() -> u32 { return dims.data[1]; }
fn NZ() -> u32 { return dims.data[2]; }
fn J () -> u32 { return dims.data[3]; }

fn colormap_blue_red(v_in: f32) -> vec4<f32> {
  let v = clamp(v_in, 0.0, 1.0);
  return vec4<f32>(v, 0.0, 1.0 - v, 1.0);
}

fn colormap_rh(v_in: f32) -> vec4<f32> {
  let v = clamp(v_in, 0.0, 1.0);

  // start = white
  let c0 = vec4<f32>(1.0, 1.0, 1.0, 1.0);
  // end   = deep blue
  let c1 = vec4<f32>(0.0, 0.0, 0.5, 1.0);

  // linear interpolation: mix(c0, c1, v)
  return c0 * (1.0 - v) + c1 * v;
}

const R_OVER_CP : f32 = 0.2857143;   // ~287/1004
const P_REF     : f32 = 100000.0;    // 1000 hPa
const EPS       : f32 = 0.622;

fn exner(p: f32) -> f32 {
  return pow(p / P_REF, R_OVER_CP);
}

fn sat_vapor_pressure(T: f32) -> f32 {
  let Tc = T - 273.15;
  // Tetens over water
  return 610.94 * exp(17.625 * Tc / (Tc + 243.04));
}

fn saturation_mixing_ratio(T: f32, p: f32) -> f32 {
  let es = sat_vapor_pressure(T);
  return EPS * es / max(p - es, 1.0);
}


@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let texSize = textureDimensions(outImg); // (nx, nz)
  if (gid.x >= texSize.x || gid.y >= texSize.y) { return; }

  let i = gid.x;  // x
  let k = gid.y;  // z
  let j = J();    // fixed y-slice

  // flat idx = i + nx*(j + ny*k)
  let idx = i + NX() * (j + NY() * k);

  let qc_val = qc[idx];

  var rgb : vec4<f32>;

  if (V.mode == 0u) {
    // theta_p view (same scaling as before)
    let t = theta_p[idx];
    let v = clamp((t + 2.0) / 4.0, 0.0, 1.0);
    rgb = colormap_blue_red(v);
  } else {
  // RH view: compute RH = qv / qs(T, p0)
  let th_tot = theta0[idx] + theta_p[idx];
  let p_cell = p0[idx];

  let Pi = exner(p_cell);
  let T  = th_tot * Pi;
  let qs = saturation_mixing_ratio(T, p_cell);

  var rh = 0.0;
  if (qs > 0.0) {
    rh = qv[idx] / qs;
  }

  let v = clamp(rh, 0.0, 1.0);
  rgb = colormap_rh(v);
}


if (V.showClouds == 1u && qc_val > 1e-6) {
    // Normalize qc relative to typical cloud range (0–5 g/kg)
    let qc_norm = clamp(qc_val / 5e-3, 0.0, 1.0);

    // Cloud brightness (white → grey as qc increases)
    let brightness = 1.0 - 0.5 * qc_norm;

    // Cloud opacity: thin clouds = faint, thick clouds = solid
    // Use a soft curve for nice visuals.
    let alpha = clamp(sqrt(qc_norm), 0.0, 1.0);

    rgb = vec4<f32>(brightness, brightness, brightness, alpha);
}




  textureStore(outImg, vec2<i32>(i32(i), i32(k)), rgb);
}
`,
      });


      const thetaPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: thetaModule, entryPoint: "main" },
      });

      // --- dims buffer for theta slice (this is what we'll tweak) ---
      const initialJSlice = Math.floor(ny / 2);
      const dimsU32 = new Uint32Array([nx, ny, nz, initialJSlice]);
      const dimsBuf = device.createBuffer({
        size: dimsU32.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(dimsBuf, 0, dimsU32);

      // 0 = theta_p, 1 = RH (rhs_qv)
      const viewUniformData = new Uint32Array([0, 1, 0, 0]);
      const viewUniformBuf = device.createBuffer({
        size: viewUniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(viewUniformBuf, 0, viewUniformData);


      const thetaBind = device.createBindGroup({
        layout: thetaPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fields.theta_p } },
          { binding: 1, resource: outView },
          { binding: 2, resource: { buffer: dimsBuf } },
          { binding: 3, resource: { buffer: fields.qc } },
          { binding: 4, resource: { buffer: fields.qv } },        // qv
          { binding: 5, resource: { buffer: fields.theta0 } },    // or fields.theta0, whichever you use
          { binding: 6, resource: { buffer: fields.p0 } },        // same
          { binding: 7, resource: { buffer: viewUniformBuf } },   // ViewUniforms
        ],
      });


      // ------------------------------------------------------------------
      // --- PARTICLES: buffers + advect compute pipeline
      // ------------------------------------------------------------------
      const N_PARTICLES = nx * nz;
      const MAX_LIFE_STEPS = 300; // tweak lifetime here

      // Particle: [x, z, life, pad] -> 4 * f32
      const particleStride = 4 * 4;
      const particleBuf = device.createBuffer({
        size: N_PARTICLES * particleStride,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      // console.log(nx, nz)

      // // initialize particles randomly over the slice
      // {
      //   const arr = new Float32Array((N_PARTICLES * particleStride) / 4);
      //   for (let i = 0; i < N_PARTICLES; i++) {
      //         const gx = i % nx;                 // grid x index 0..nx-1
      //     const gz = Math.floor(i / nx);     // grid z index 0..nz-1

      //     const base = 4 * i;
      //           // arr[base + 0] = Math.random() * (nx - 1); // x index
      //           // arr[base + 1] = Math.random() * (nz - 1); // z index
      //           arr[base + 0] = gx + 0.5; // x index
      //           arr[base + 1] = gz+ 0.5; // z index
      //           console.log(gx, gz)
      //               arr[base + 2] = MAX_LIFE_STEPS;   // life
      //     arr[base + 3] = 0.0;  // pad
      //         }
      //   device.queue.writeBuffer(particleBuf, 0, arr);
      // }
      // initialize particles on a uniform grid over the x–z slice
      // choose a logical grid resolution for particles
      // npx * npz should be >= N_PARTICLES
      const aspect = nx / nz;
      let npx = Math.max(1, Math.round(Math.sqrt(N_PARTICLES * aspect)));
      let npz = Math.max(1, Math.ceil(N_PARTICLES / npx));

      // spacing in index space
      const dxp = nx / npx; // width of one particle cell in x
      const dzp = nz / npz; // height of one particle cell in z

      {
        const arr = new Float32Array((N_PARTICLES * particleStride) / 4);

        for (let i = 0; i < N_PARTICLES; i++) {
          const gx = i % npx;             // 0 .. npx-1
          const gz = Math.floor(i / npx); // 0 .. npz-1 (until we run out of particles)

          const base = 4 * i;

          // put the particle at the center of its logical cell,
          // mapped to [0, nx) x [0, nz)
          arr[base + 0] = (gx + 0.5) * dxp; // x in [0, nx)
          arr[base + 1] = (gz + 0.5) * dzp; // z in [0, nz)
          arr[base + 2] = MAX_LIFE_STEPS;   // life
          arr[base + 3] = 0.0;              // pad
        }

        device.queue.writeBuffer(particleBuf, 0, arr);
      }



      // uniforms: N (as u32) + padding to 16 bytes
      // PartUniforms: [N, frame, maxLifeSteps, pad]
      const partUniformData = new Uint32Array([N_PARTICLES, 0, MAX_LIFE_STEPS, 0]);
      const partUniformBuf = device.createBuffer({
        size: partUniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(partUniformBuf, 0, partUniformData);



      const advectModule = device.createShaderModule({
        code: /* wgsl */ `
struct Dims { data: array<u32> };

struct Particle {
  x: f32,
  z: f32,
  life: f32,
  pad: f32,
};

struct PartUniforms {
  N: u32,
  frame: u32,
  maxLifeSteps: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<storage, read> u : array<f32>;
@group(0) @binding(2) var<storage, read> v : array<f32>;
@group(0) @binding(3) var<storage, read> w : array<f32>;
@group(0) @binding(4) var<storage, read> dims : Dims;
@group(0) @binding(5) var<uniform> U : PartUniforms;

fn NX() -> u32 { return dims.data[0]; }
fn NY() -> u32 { return dims.data[1]; }
fn NZ() -> u32 { return dims.data[2]; }
fn J () -> u32 { return dims.data[3]; }

// simple hash-based random in [0,1)
fn rand01(i: u32, frame: u32) -> f32 {
  var n = i ^ (frame * 747796405u + 2891336453u);
  n = (n ^ (n >> 16u)) * 2246822519u;
  return f32(n & 0x00FFFFFFu) / f32(0x01000000u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idxP = gid.x;
  let Np = U.N;
  if (idxP >= Np) { return; }

  var p = particles[idxP];

  let nx = f32(NX());
  let nz = f32(NZ());

  let ix = u32(clamp(p.x, 0.0, nx - 1.0));
  let iz = u32(clamp(p.z, 0.0, nz - 1.0));

  let idx = ix + NX() * (J() + NY() * iz);

  let uvel = u[idx];
  let wvel = w[idx];

  let dt = 0.5;

  // simple forward Euler advection in slice space
let dxp = uvel * dt;
let dzp = wvel * dt;

p.x = p.x + dxp;
p.z = p.z + dzp;

  // wrap around domain so particles don't leave
  // NOT CORRECT BEHAVIOR - PLACEHOLDER UNTIL 3D PARTICLE SIMULATION AND WE'RE NOT DOING JUST A SLICE
  if (p.x < 0.0) { p.x = p.x + nx; }
  if (p.x >= nx) { p.x = p.x - nx; }
  // if (p.z < 0.0) { p.z = p.z + nz; }
  // if (p.z >= nz) { p.z = p.z - nz; }

  // ---- distance-based life decay + small random part ----
let dist = sqrt(dxp * dxp + dzp * dzp);  // distance moved this step

// tune these two constants however you like:
let distScale = 15.0;                     // how much distance eats life
let baseCost  = 0.1;                     // minimum decay per step

// small random extra in [-noiseAmp/2, +noiseAmp/2]
let noiseAmp = 10.0;
let jitter = noiseAmp * (rand01(idxP, U.frame) - 0.5);

let decay = baseCost + distScale * dist + jitter;

p.life = p.life - decay;

  // respawn when life is over
if (p.life <= 0.0) {
  // Rebuild the same logical grid layout using idxP
  let Np = U.N;

  // near-square grid
  let rootN = sqrt(f32(Np));
  var npx = u32(rootN);
  if (npx < 1u) {
    npx = 1u;
  }
  let npz = (Np + npx - 1u) / npx;

  let gx = idxP % npx;
  let gz = idxP / npx;

  let dxp = nx / f32(npx);
  let dzp = nz / f32(npz);

  p.x = (f32(gx) + 0.5) * dxp;
  p.z = (f32(gz) + 0.5) * dzp;
  p.life = f32(U.maxLifeSteps);
}

  particles[idxP] = p;
}
`,
      });



      const advectBGL = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 5,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" },
          },
        ],
      });

      const advectPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [advectBGL] }),
        compute: { module: advectModule, entryPoint: "main" },
      });

      const advectBind = device.createBindGroup({
        layout: advectBGL,
        entries: [
          { binding: 0, resource: { buffer: particleBuf } },
          { binding: 1, resource: { buffer: fields.u } },
          { binding: 2, resource: { buffer: fields.v } },
          { binding: 3, resource: { buffer: fields.w } },
          { binding: 4, resource: { buffer: dimsBuf } },
          { binding: 5, resource: { buffer: partUniformBuf } },
        ],
      });


      // ------------------------------------------------------------------
      // TWEAKPANE: control jSlice (dimsU32[3])
      // ------------------------------------------------------------------
      if (paneRef.current) {
        const pane = new Pane({ container: paneRef.current });
        const controls = {
          jSlice: initialJSlice,
          viewField: "theta" as "theta" | "rh",
          showClouds: true,
        };


        pane
          .addBinding(controls, "jSlice", {
            min: 0,
            max: ny - 1,
            step: 1,
            label: "Y slice (j)",
          })
          .on("change", (ev: { value: number }) => {
            const j = ev.value | 0;
            dimsU32[3] = j;
            device.queue.writeBuffer(dimsBuf, 0, dimsU32);
          });

        pane
          .addBinding(controls, "viewField", {
            label: "Field",
            options: {
              "Theta (θ')": "theta",
              "RH (qv/qs)": "rh",
            },
          })
          .on("change", (ev: { value: "theta" | "rh" }) => {
            viewUniformData[0] = ev.value === "rh" ? 1 : 0;
            device.queue.writeBuffer(viewUniformBuf, 0, viewUniformData);
          });

        pane.addBinding(controls, "showClouds", { label: "Show Clouds" })
          .on("change", (ev: { value: boolean }) => {
            viewUniformData[1] = ev.value ? 1 : 0;
            device.queue.writeBuffer(viewUniformBuf, 0, viewUniformData);
          });


      }

      // ------------------------------------------------------------------
      // RENDER (fullscreen) for theta slice
      // ------------------------------------------------------------------
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "opaque" });

      const renderModule = device.createShaderModule({
        code: /* wgsl */ `
struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0)
  );
  let p = positions[vi];
  var out : VSOut;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = 0.5 * (p + vec2<f32>(1.0, 1.0));
  return out;
}

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var img  : texture_2d<f32>;

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(img, samp, in.uv, 0.0);
}
`,
      });

      const sampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
      const renderPipeline = await device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module: renderModule, entryPoint: "vs_main" },
        fragment: { module: renderModule, entryPoint: "fs_main", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });

      const renderBind = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: outView },
        ],
      });

      // ------------------------------------------------------------------
      // --- PARTICLES: render pipeline
      // ------------------------------------------------------------------
      const particleRenderModule = device.createShaderModule({
        code: /* wgsl */ `
struct Dims { data: array<u32> };

struct Particle {
  x: f32,
  z: f32,
  life: f32,
  pad: f32,
};


struct PartUniforms {
  N: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};


struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) whiteness : f32,
};

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<storage, read> vField : array<f32>;
@group(0) @binding(2) var<storage, read> dims : Dims;
@group(0) @binding(3) var<uniform> U : PartUniforms;

fn NX() -> u32 { return dims.data[0]; }
fn NY() -> u32 { return dims.data[1]; }
fn NZ() -> u32 { return dims.data[2]; }
fn J () -> u32 { return dims.data[3]; }

@vertex
fn vs_main(
  @builtin(vertex_index) vi : u32,
  @builtin(instance_index) inst : u32
) -> VSOut {
  let Np = U.N;
  var out : VSOut;

  if (inst >= Np) {
    out.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.whiteness = 0.0;
    return out;
  }

  let nx = f32(NX());
  let nz = f32(NZ());

  // Particle center in NDC
  let p = particles[inst];
  let x_ndc = (p.x / nx) * 2.0 - 1.0;
  let z_ndc = (p.z / nz) * 2.0 - 1.0;
  let center = vec2<f32>(x_ndc, z_ndc);

  // Size of the quad in NDC (tweak this for bigger/smaller particles)
  let s = 0.005;

  // 6 vertices forming 2 triangles of a quad
  // (−s,−s), ( s,−s), (−s, s),
  // (−s, s), ( s,−s), ( s, s)
  var offset : vec2<f32>;
  switch vi {
    case 0u: { offset = vec2<f32>(-s, -s); }
    case 1u: { offset = vec2<f32>( s, -s); }
    case 2u: { offset = vec2<f32>(-s,  s); }
    case 3u: { offset = vec2<f32>(-s,  s); }
    case 4u: { offset = vec2<f32>( s, -s); }
    default: { offset = vec2<f32>( s,  s); }
  }

  let pos2 = center + offset;
  out.pos = vec4<f32>(pos2, 0.0, 1.0);

  // color from v (same idea as before)
  let ix = u32(clamp(p.x, 0.0, nx - 1.0));
  let iz = u32(clamp(p.z, 0.0, nz - 1.0));
  let idx = ix + NX() * (J() + NY() * iz);
  let v = vField[idx];

  let scale = 0.5;
  out.whiteness = clamp(0.5 + scale * v, 0.0, 1.0);

  return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  let c = in.whiteness;
  return vec4<f32>(c, c, c, 1.0);
}
`,
      });


      const particleRenderPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: particleRenderModule, entryPoint: "vs_main" },
        fragment: { module: particleRenderModule, entryPoint: "fs_main", targets: [{ format }] },
        primitive: { topology: "triangle-list" },

      });

      const particleRenderBind = device.createBindGroup({
        layout: particleRenderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: particleBuf } },
          { binding: 1, resource: { buffer: fields.v } },
          { binding: 2, resource: { buffer: dimsBuf } },
          { binding: 3, resource: { buffer: partUniformBuf } },
        ],
      });

      // --- animation flags/time ---
      let running = true;
      let last = performance.now();
      let accumulator = 0;
      const maxFrameDt = 1 / 15;
      const subDt = dt;
      let frameIndex = 0;

      async function frame(now: number) {
        if (!running) return;

        let dtSec = (now - last) / 1000;
        last = now;
        dtSec = Math.min(dtSec, maxFrameDt);
        accumulator += dtSec;
        setAccumulatedTime((accTime) => accTime + dtSec);

        device.pushErrorScope("validation");

        frameIndex++;
        partUniformData[1] = frameIndex; // update frame
        device.queue.writeBuffer(partUniformBuf, 0, partUniformData);


        const encoder = device.createCommandEncoder();

        // physics step
        rk2.step(encoder, subDt);

        // --- PARTICLES: advect on slice using u, w at current j ---
        {
          const cpass = encoder.beginComputePass();
          cpass.setPipeline(advectPipeline);
          cpass.setBindGroup(0, advectBind);
          const wg = Math.ceil(N_PARTICLES / 256);
          cpass.dispatchWorkgroups(wg, 1, 1);
          cpass.end();
        }

        // theta slice to texture
        {
          const cpass = encoder.beginComputePass();
          cpass.setPipeline(thetaPipeline);
          cpass.setBindGroup(0, thetaBind);
          const wgX = Math.ceil(nx / 16);
          const wgY = Math.ceil(nz / 16);
          cpass.dispatchWorkgroups(wgX, wgY, 1);
          cpass.end();
        }

        const view = context!.getCurrentTexture().createView();
        {
          const rpass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view,
                clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1 },
                loadOp: "clear",
                storeOp: "store",
              },
            ],
          });

          // base slice
          rpass.setPipeline(renderPipeline);
          rpass.setBindGroup(0, renderBind);
          rpass.draw(3);

          // --- PARTICLES: draw on top ---
          rpass.setPipeline(particleRenderPipeline);
          rpass.setBindGroup(0, particleRenderBind);
          rpass.draw(6, N_PARTICLES, 0, 0); // 6 verts per particle, N_PARTICLES instances

          rpass.end();
        }

        device.queue.submit([encoder.finish()]);

        const err = await device.popErrorScope();
        if (err) console.error("WebGPU validation error (frame):", err.message);

        requestAnimationFrame(frame);
      }

      requestAnimationFrame(frame);
    };

    init();
  }, []);

  return (
    <main className="flex min-h-screen items-center justify-center bg-black">
      <div className="flex h-[600px]">
        {/* canvas + overlay */}
        <div className="relative">
          <h1 className="absolute top-2 left-2 text-black z-10">
            Time: {accumulatedTime.toFixed(0)}s
          </h1>
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="border border-neutral-800"
          />
        </div>

        {/* tweakpane on the right, full height of this row */}
        <div
          ref={paneRef}
          className="ml-4 w-64 h-full bg-neutral-900 rounded-md"
        />
      </div>
    </main>
  );
}
