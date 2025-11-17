"use client";

import { useEffect, useRef, useState } from "react";
import { Pane } from "tweakpane";
import { initBoussinesq3D } from "./init_boussinesq";
import { makeStepRK2 } from "./step_rk2";
import { makeComputeRhs } from "./compute_rhs";

// Assume fields.w is your GPUBuffer
async function debugPrintBufferMax(device: GPUDevice, buffer: GPUBuffer, count: number) {
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

      const dt = 0.2;

      // ------------------------------------------------------------------
      // COMPUTE PIPELINE: theta slice to outTex
      // ------------------------------------------------------------------
      const thetaModule = device.createShaderModule({
        code: /* wgsl */ `
struct Dims { data: array<u32> }; // [0]=nx, [1]=ny, [2]=nz, [3]=j

@group(0) @binding(0) var<storage, read> theta_p : array<f32>;
@group(0) @binding(1) var outImg : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> dims : Dims;
@group(0) @binding(3) var<storage, read> qc : array<f32>;

fn NX() -> u32 { return dims.data[0]; }
fn NY() -> u32 { return dims.data[1]; }
fn NZ() -> u32 { return dims.data[2]; }
fn J () -> u32 { return dims.data[3]; }

fn colormap_blue_red(v_in: f32) -> vec3<f32> {
  let v = clamp(v_in, 0.0, 1.0);
  return vec3<f32>(v, 0.0, 1.0 - v);
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

  let t  = theta_p[idx];
  let qc_val = qc[idx];

  let v = clamp((t + 2.0) / 4.0, 0.0, 1.0);
  var rgb = colormap_blue_red(v);

  if (qc_val > 1e-4) {
    rgb = vec3<f32>(1.0, 1.0, 0.0);
  }

  textureStore(outImg, vec2<i32>(i32(i), i32(k)), vec4<f32>(rgb, 1.0));
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

      const thetaBind = device.createBindGroup({
        layout: thetaPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fields.theta_p } },
          { binding: 1, resource: outView },
          { binding: 2, resource: { buffer: dimsBuf } },
          { binding: 3, resource: { buffer: fields.qc } },
        ],
      });

      // ------------------------------------------------------------------
      // --- PARTICLES: buffers + advect compute pipeline
      // ------------------------------------------------------------------
      const N_PARTICLES = 20000;

      // particle positions in grid coords (x, z) on the slice
      const particleStride = 2 * 4; // 2 * f32
      const particleBuf = device.createBuffer({
        size: N_PARTICLES * particleStride,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      // initialize particles randomly over the slice
      {
        const arr = new Float32Array((N_PARTICLES * particleStride) / 4);
        for (let i = 0; i < N_PARTICLES; i++) {
          arr[2 * i + 0] = Math.random() * (nx - 1); // x index
          arr[2 * i + 1] = Math.random() * (nz - 1); // z index
        }
        device.queue.writeBuffer(particleBuf, 0, arr);
      }

// uniforms: N (as u32) + padding to 16 bytes
const partUniformData = new Uint32Array([N_PARTICLES, 0, 0, 0]);
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
};

struct PartUniforms {
  N: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
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

  // NOTE: dt is baked in from the host; you can add it to uniforms if you want
  let dt = 0.2;

  // simple forward Euler advection in slice space
  p.x = p.x + uvel * dt;
  p.z = p.z + wvel * dt;

  // wrap around domain so particles don't leave
  if (p.x < 0.0) { p.x = p.x + nx; }
  if (p.x >= nx) { p.x = p.x - nx; }
  if (p.z < 0.0) { p.z = p.z + nz; }
  if (p.z >= nz) { p.z = p.z - nz; }

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

      async function frame(now: number) {
        if (!running) return;

        let dtSec = (now - last) / 1000;
        last = now;
        dtSec = Math.min(dtSec, maxFrameDt);
        accumulator += dtSec;
        setAccumulatedTime((accTime) => accTime + dtSec);

        device.pushErrorScope("validation");

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
          <h1 className="absolute top-2 left-2 text-white z-10">
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
