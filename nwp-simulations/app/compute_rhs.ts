// compute_rhs.ts
import type { SimDims, FieldBuffers, Params } from "./init_boussinesq";
import { makeAdvectScalar } from "./advect_scalar";
import { makeDiffuseVelocity } from "./diffuse_velocity";
import { makeDiffuseScalar } from "./diffuse_scalar";
import { makeRadiativeCooling } from "./radiative_cooling";
import { makeSurfaceRelax } from "./surface_relax";

export type BuildRHSIO = {
    state: {
        theta_p: GPUBuffer;
        qv: GPUBuffer;
        qc: GPUBuffer;
        u: GPUBuffer;
        v: GPUBuffer;
        w: GPUBuffer;
    };
    out: {
        rhs_w: GPUBuffer;
        rhs_theta_p: GPUBuffer;
        rhs_qv: GPUBuffer;
        rhs_qc: GPUBuffer;
        rhs_u: GPUBuffer; rhs_v: GPUBuffer;
    };
};

export type ComputeRhsArtifacts = {
    buildRHS: (pass: GPUComputePassEncoder, io?: Partial<BuildRHSIO>) => void;
    defaults: BuildRHSIO;
    resources: {
        pipeline: GPUComputePipeline;    // buoyancy/damping/theta-cooling
        rhsUniformBuf: GPUBuffer;
        workgroups: number;
        advect: ReturnType<typeof makeAdvectScalar>;
    };
};

export function makeComputeRhs(opts: {
    device: GPUDevice;
    fields: FieldBuffers & { bg_thermo: GPUBuffer };
    dims: SimDims;
    params: Params;
}): ComputeRhsArtifacts {
    const { device, fields, dims, params } = opts;
    const { nx, ny, nz } = dims;

    // --- existing buoyancy+damp+theta-cooling WGSL unchanged except Uniforms struct already includes tau_w ---
    const computeRhsModule = device.createShaderModule({
        code: /* wgsl */`
struct Uniforms {
  N: u32,
  offTheta0: u32,
  offDthdz: u32,
  offQvbg: u32,
  g: f32,
  tau_w: f32,
  _pad: vec2<f32>,
};

@group(0) @binding(0) var<storage, read>        theta_p      : array<f32>;
@group(0) @binding(1) var<storage, read>        qv           : array<f32>;
@group(0) @binding(2) var<storage, read>        qc           : array<f32>;
@group(0) @binding(3) var<storage, read>        w_in         : array<f32>;
@group(0) @binding(4) var<storage, read_write>  rhs_w        : array<f32>;
@group(0) @binding(5) var<storage, read_write>  rhs_theta_p  : array<f32>;
@group(0) @binding(6) var<storage, read>        bg           : array<f32>;
@group(0) @binding(7) var<uniform>              U            : Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.N) { return; }

  let thp   = theta_p[i];
  let qv_i  = qv[i];
  let qc_i  = qc[i];
  let w_i   = w_in[i];

  let th0   = bg[U.offTheta0 + i];
  let dthdz = bg[U.offDthdz + i];
  let qvbg  = bg[U.offQvbg  + i];

  // buoyancy
  let b = U.g * ( (thp / th0) + 0.61 * (qv_i - qvbg) - qc_i );
  rhs_w[i] = rhs_w[i] + b;

  // damping: rhs_w += -(w/tau)
  rhs_w[i] = rhs_w[i] - (w_i / U.tau_w);

  // theta' background cooling
  rhs_theta_p[i] = rhs_theta_p[i] + (-w_i * dthdz);
}
`});

    const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: computeRhsModule, entryPoint: "main" },
    });

    // uniforms for buoyancy kernel
    const N = nx * ny * nz;
    const offTheta0 = 0, offDthdz = N, offQvbg = 2 * N;

    const rhsUniformsAB = new ArrayBuffer(48);
    const u32 = new Uint32Array(rhsUniformsAB);
    const f32 = new Float32Array(rhsUniformsAB);
    u32[0] = N >>> 0;
    u32[1] = offTheta0 >>> 0;
    u32[2] = offDthdz >>> 0;
    u32[3] = offQvbg >>> 0;
    f32[4] = (params.g ?? 9.81);
    f32[5] = (params.tau_damp_w ?? 300.0);

    const rhsUniformBuf = device.createBuffer({ size: rhsUniformsAB.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rhsUniformBuf, 0, rhsUniformsAB);

    // defaults (now include u,v as well as qv/qc RHS)
    const defaults: BuildRHSIO = {
        state: {
            theta_p: fields.theta_p,
            qv: fields.qv,
            qc: fields.qc,
            u: fields.u,
            v: fields.v,
            w: fields.w,
        },
        out: {
            rhs_w: fields.rhs_w,
            rhs_theta_p: fields.rhs_theta_p,
            rhs_qv: fields.rhs_qv,
            rhs_qc: fields.rhs_qc,
            rhs_u: fields.rhs_u,
            rhs_v: fields.rhs_v,
        },
    };

    const workgroups = Math.ceil(N / 256);
    const advect = makeAdvectScalar({ device, dims, params });
    const diffuseVel = makeDiffuseVelocity({ device, dims, params });
    const diffTheta = makeDiffuseScalar({ device, dims, coeff: params.kappa ?? 0.0 });
const diffMoist = makeDiffuseScalar({ device, dims, coeff: params.Dq ?? 0.0 });
const radCool = makeRadiativeCooling({ device, dims, params });
const surfRelax = makeSurfaceRelax({ device, dims, params });

    // helper to build buoyancy bindgroup quickly
    const makeBG_buoy = (io: BuildRHSIO) =>
        device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: io.state.theta_p } },
                { binding: 1, resource: { buffer: io.state.qv } },
                { binding: 2, resource: { buffer: io.state.qc } },
                { binding: 3, resource: { buffer: io.state.w } },
                { binding: 4, resource: { buffer: io.out.rhs_w } },
                { binding: 5, resource: { buffer: io.out.rhs_theta_p } },
                { binding: 6, resource: { buffer: (fields as any).bg_thermo } },
                { binding: 7, resource: { buffer: rhsUniformBuf } },
            ],
        });

    function buildRHS(pass: GPUComputePassEncoder, ioPartial?: Partial<BuildRHSIO>) {
        // nested merge with defaults
        const io: BuildRHSIO = {
            state: {
                theta_p: ioPartial?.state?.theta_p ?? defaults.state.theta_p,
                qv: ioPartial?.state?.qv ?? defaults.state.qv,
                qc: ioPartial?.state?.qc ?? defaults.state.qc,
                u: ioPartial?.state?.u ?? defaults.state.u,
                v: ioPartial?.state?.v ?? defaults.state.v,
                w: ioPartial?.state?.w ?? defaults.state.w,
            },
            out: {
                rhs_w: ioPartial?.out?.rhs_w ?? defaults.out.rhs_w,
                rhs_theta_p: ioPartial?.out?.rhs_theta_p ?? defaults.out.rhs_theta_p,
                rhs_qv: ioPartial?.out?.rhs_qv ?? defaults.out.rhs_qv,
                rhs_qc: ioPartial?.out?.rhs_qc ?? defaults.out.rhs_qc,
                rhs_u: ioPartial?.out?.rhs_u ?? defaults.out.rhs_u,
                rhs_v: ioPartial?.out?.rhs_v ?? defaults.out.rhs_v,
            },
        };

        // ---- 1) buoyancy + damping + theta bg cooling
        {
            const bg = makeBG_buoy(io);
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(workgroups);
        }

        // ---- 2) velocity advection (first-order upwind, conservative)
        advect.dispatch(pass, {
            phi: io.state.u,
            u: io.state.u, v: io.state.v, w: io.state.w,
            rhs_out: io.out.rhs_u,
        });
        advect.dispatch(pass, {
            phi: io.state.v,
            u: io.state.u, v: io.state.v, w: io.state.w,
            rhs_out: io.out.rhs_v,
        });
        advect.dispatch(pass, {
            phi: io.state.w,
            u: io.state.u, v: io.state.v, w: io.state.w,
            rhs_out: io.out.rhs_w,   // add dv_adv to existing rhs_w (with buoyancy/damping)
        });


        // ---- 3) scalar advection (ordered right after buoyancy in the SAME pass)
        advect.dispatch(pass, {
            phi: io.state.theta_p,
            u: io.state.u, v: io.state.v, w: io.state.w,
            rhs_out: io.out.rhs_theta_p,
        });
        advect.dispatch(pass, {
            phi: io.state.qv,
            u: io.state.u, v: io.state.v, w: io.state.w,
            rhs_out: io.out.rhs_qv,
        });
        advect.dispatch(pass, {
            phi: io.state.qc,
            u: io.state.u, v: io.state.v, w: io.state.w,
            rhs_out: io.out.rhs_qc,
        });

        // ---- 4) velocity diffusion (ν ∇² u,v,w)
        diffuseVel.dispatch(pass, {
            u: io.state.u,
            v: io.state.v,
            w: io.state.w,
            rhs_u: io.out.rhs_u,
            rhs_v: io.out.rhs_v,
            rhs_w: io.out.rhs_w,
        });

        // ---- 5) scalar diffusion
diffTheta.dispatch(pass, {
  phi: io.state.theta_p,
  rhs_out: io.out.rhs_theta_p,
});

diffMoist.dispatch(pass, {
  phi: io.state.qv,
  rhs_out: io.out.rhs_qv,
});
diffMoist.dispatch(pass, {
  phi: io.state.qc,
  rhs_out: io.out.rhs_qc,
});

// 6 -- radiative cooling
radCool.dispatch(pass, {
  theta_p: io.state.theta_p,
  rhs_theta_p: io.out.rhs_theta_p,
});

// 7 - add moisture and heat at the bottom
surfRelax.dispatch(pass, {
  theta0: fields.theta0,                 // background
  theta_p: io.state.theta_p,             // current θ′
  qv: io.state.qv,                       // current qv
  theta_target: fields.theta_surf_target,
  qv_target:    fields.qv_surf_target,
  rhs_theta_p:  io.out.rhs_theta_p,
  rhs_qv:       io.out.rhs_qv,
});
    }

    return {
        buildRHS,
        defaults,
        resources: { pipeline, rhsUniformBuf, workgroups, advect },
    };
}
