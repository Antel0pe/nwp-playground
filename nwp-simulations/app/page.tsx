"use client";

import { useEffect, useRef, useState } from "react";
import { initBoussinesq3D } from "./init_boussinesq";
import { makeStepRK2 } from "./step_rk2";
import { makeComputeRhs } from "./compute_rhs";


// Assume fields.w is your GPUBuffer
async function debugPrintBufferMax(device: GPUDevice, buffer: GPUBuffer, count: number) {
  // 1. Create a temporary read buffer with MAP_READ usage
  const readBuffer = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // 2. Encode copy from GPU-only buffer to this readable buffer
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size);
  device.queue.submit([encoder.finish()]);

  // 3. Map it for reading
  await readBuffer.mapAsync(GPUMapMode.READ);
  const arr = new Float32Array(readBuffer.getMappedRange());

  // 4. Inspect
  const maxVal = Math.max(...arr);
  console.log("Max value in buffer:", maxVal);
  console.log("First few values:", arr.slice(0, 10));

  // 5. Clean up
  readBuffer.unmap();
}


export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
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
      device.lost.then(info => {
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
  computeRhs, // ← pass the artifact, not the bare pipeline/bindgroup
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

fn NX() -> u32 { return dims.data[0]; }
fn NY() -> u32 { return dims.data[1]; }
fn NZ() -> u32 { return dims.data[2]; }
fn J () -> u32 { return dims.data[3]; }

fn colormap_blue_red(v_in: f32) -> vec3<f32> {
  let v = clamp(v_in, 0.0, 1.0);

  // Simple blue → red gradient, through purple
  // v = 0   -> (0, 0, 1) blue
  // v = 0.5 -> (0.5, 0, 0.5) purple
  // v = 1   -> (1, 0, 0) red
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
  let t = theta_p[idx];

  // quick viz
  let v = clamp(t / 1.8, 0.0, 1.0);
  let rgb = colormap_blue_red(v);
  textureStore(outImg, vec2<i32>(i32(i), i32(k)), vec4<f32>(rgb, 1.0));
}
`,
      });

      const thetaPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: thetaModule, entryPoint: "main" },
      });

      // dims buffer for theta slice
      const jSlice = Math.floor(ny / 2);
      const dimsU32 = new Uint32Array([nx, ny, nz, jSlice]);
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
        ],
      });

      // ------------------------------------------------------------------
      // RENDER (fullscreen)
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


      // --- animation flags/time ---
      let running = true;
      let last = performance.now();
      let accumulator = 0;
      const maxFrameDt = 1 / 15;   // clamp big frame gaps (s)
      const subDt = dt;            // or choose a smaller physics substep for CFL

      async function frame(now: number) {
        if (!running) return;

        // await debugPrintBufferMax(device, fields.w, nx * ny * nz);


        // --- timekeeping ---
        let dtSec = (now - last) / 1000;
        last = now;
        dtSec = Math.min(dtSec, maxFrameDt);
        accumulator += dtSec;
        setAccumulatedTime((accTime) => accTime + dtSec)

        // Optional: multiple substeps if physics needs smaller dt than render
        // E.g., while (accumulator >= subDt) { step physics with subDt; accumulator -= subDt; }
        // Here we’ll just do one step per frame; switch to the loop above if needed.

        device.pushErrorScope("validation");

        // One encoder for the whole frame
        const encoder = device.createCommandEncoder();

        // 1) physics step (records compute passes inside step)
        rk2.step(encoder, subDt); // change to subDt if you enable substepping

        // 2) theta slice -> texture (compute)
        {
          const cpass = encoder.beginComputePass();
          cpass.setPipeline(thetaPipeline);
          cpass.setBindGroup(0, thetaBind);
          const wgX = Math.ceil(nx / 16);
          const wgY = Math.ceil(nz / 16);
          cpass.dispatchWorkgroups(wgX, wgY, 1);
          cpass.end();
        }

        // 3) render
        const view = context!.getCurrentTexture().createView();
        {
          const rpass = encoder.beginRenderPass({
            colorAttachments: [{
              view,
              clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1 },
              loadOp: "clear",
              storeOp: "store",
            }],
          });
          rpass.setPipeline(renderPipeline);
          rpass.setBindGroup(0, renderBind);
          rpass.draw(3);
          rpass.end();
        }

        device.queue.submit([encoder.finish()]);

        // Pop the scope we pushed this frame
        const err = await device.popErrorScope();
        if (err) console.error("WebGPU validation error (frame):", err.message);

        requestAnimationFrame(frame);
      }

      // Start the loop once everything is initialized
      requestAnimationFrame(frame);

      // Optional: pause when tab hidden (saves GPU)
      // document.addEventListener("visibilitychange", () => {
      //   if (document.hidden) {
      //     running = false;
      //   } else if (!running) {
      //     running = true;
      //     last = performance.now();
      //     requestAnimationFrame(frame);
      //   }
      // });

      // // Optional: handle canvas resize for crisp rendering
      // function resizeCanvas() {
      //   const dpr = Math.min(window.devicePixelRatio || 1, 2);
      //   const w = Math.floor(canvas.clientWidth * dpr);
      //   const h = Math.floor(canvas.clientHeight * dpr);
      //   if (canvas.width !== w || canvas.height !== h) {
      //     canvas.width = w;
      //     canvas.height = h;
      //   }
      // }
      // window.addEventListener("resize", resizeCanvas);
      // resizeCanvas();

    };

    init();
  }, []);

return (
  <main className="flex min-h-screen items-center justify-center bg-black">
    <div className="relative">
      {/* overlay */}
      <h1 className="absolute top-2 left-2 text-white z-10">
        Time: {accumulatedTime.toFixed(0)}s
      </h1>

      {/* canvas */}
      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        className="border border-neutral-800"
      />
    </div>
  </main>
);
}
