"use client";

import { useEffect, useRef } from "react";

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

      const context = canvas.getContext("webgpu");
      if (!context) {
        console.error("Failed to get WebGPU context.");
        return;
      }

      const format = navigator.gpu.getPreferredCanvasFormat();

      context.configure({
        device,
        format,
        alphaMode: "opaque",
      });

      const shaderModule = device.createShaderModule({
        label: "hello-webgpu-shader",
        code: /* wgsl */ `
          @vertex
          fn vs_main(@builtin(vertex_index) vertexIndex : u32)
              -> @builtin(position) vec4<f32> {
            var positions = array<vec2<f32>, 3>(
              vec2<f32>(0.0, 0.5),
              vec2<f32>(-0.5, -0.5),
              vec2<f32>(0.5, -0.5)
            );
            let xy = positions[vertexIndex];
            return vec4<f32>(xy, 0.0, 1.0);
          }

          @fragment
          fn fs_main() -> @location(0) vec4<f32> {
            // Hello World: cyan-ish triangle
            return vec4<f32>(0.0, 0.8, 0.9, 1.0);
          }
        `,
      });

      const pipeline = await device.createRenderPipelineAsync({
        label: "hello-webgpu-pipeline",
        layout: "auto",
        vertex: {
          module: shaderModule,
          entryPoint: "vs_main",
        },
        fragment: {
          module: shaderModule,
          entryPoint: "fs_main",
          targets: [{ format }],
        },
        primitive: {
          topology: "triangle-list",
        },
      });

      const render = () => {
        const encoder = device.createCommandEncoder();
        const view = context.getCurrentTexture().createView();

        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view,
              clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1.0 },
              loadOp: "clear",
              storeOp: "store",
            },
          ],
        });

        pass.setPipeline(pipeline);
        pass.draw(3, 1, 0, 0);
        pass.end();

        device.queue.submit([encoder.finish()]);
      };

      render();
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
