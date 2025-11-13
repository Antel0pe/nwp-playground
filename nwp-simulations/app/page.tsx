"use client";

import { useEffect, useRef } from "react";
import { initBoussinesq3D } from "./init_boussinesq";

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

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

      const context = canvas.getContext("webgpu");
      if (!context) {
        console.error("Failed to get WebGPU context.");
        return;
      }

      const { fields, params, dims } = await initBoussinesq3D(device);
      const { nx, ny, nz } = dims;

      // Output texture written by compute & sampled by render
      const outTex = device.createTexture({
        size: { width: nx, height: nz },
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      });
      const outView = outTex.createView();

      // ---- Compute pipeline for theta_p slice ----
      const thetaModule = device.createShaderModule({
        code: /* wgsl */ `
struct Dims { data: array<u32> }; // [0]=nx, [1]=ny, [2]=nz, [3]=j

@group(0) @binding(0) var<storage, read> theta_p : array<f32>;
@group(0) @binding(1) var outImg : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> dims : Dims;

fn NX() -> u32 { return dims.data[0]; }
fn NY() -> u32 { return dims.data[1]; }
fn NZ() -> u32 { return dims.data[2]; }
fn J () -> u32 { return dims.data[3]; } // <- fixed y-plane

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  // outImg is sized (nx, nz)
  let texSize = textureDimensions(outImg);
  if (gid.x >= texSize.x || gid.y >= texSize.y) { return; }

  let i = gid.x;         // x
  let k = gid.y;         // z
  let j = J();           // fixed y

  // flat index: i + nx*(j + ny*k)
  let idx = i + NX() * (j + NY() * k);
  let t = theta_p[idx];

  // visualize (adjust range if you like)
  let v = clamp(t / 1.8, 0.0, 1.0);
  textureStore(outImg, vec2<i32>(i32(i), i32(k)), vec4<f32>(v, v, v, 1.0));
}

`,
      });

      const thetaPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: thetaModule, entryPoint: "main" },
      });

      // dims via read-only STORAGE buffer
      const jSlice = Math.floor(ny / 2)
      const dimsU32 = new Uint32Array([nx, ny, nz, jSlice]); // k=8 ≈ 800 m
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

      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "opaque" });

      // ---- Render pipeline (fullscreen triangle) ----
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

      const renderOnce = async () => {
        const encoder = device.createCommandEncoder();

        // Compute pass (theta slice)
        const cpass = encoder.beginComputePass();
        cpass.setPipeline(thetaPipeline);
        cpass.setBindGroup(0, thetaBind);
        const wgX = Math.ceil(nx / 16);
        const wgY = Math.ceil(ny / 16);
        console.log("THETA dispatch", wgX, wgY, "dims", nx, ny, nz);
        cpass.dispatchWorkgroups(wgX, wgY, 1);
        cpass.end();

        // Render full screen
        const view = context.getCurrentTexture().createView();
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

        device.queue.submit([encoder.finish()]);

        const err = await device.popErrorScope();
        if (err) console.error("WebGPU validation error (frame):", err.message);
      };

      renderOnce();
    };

    init();
  }, []);

  return (
    <main className="flex min-h-screen items-center justify-center bg-black">
      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        className="border border-neutral-800"
      />
    </main>
  );
}
