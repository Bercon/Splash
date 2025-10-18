export const mlsmpmParticleStructSize = 80

export class MLSMPMSimulator {
    cellStructSize = 16;
    realBoxSizeBuffer: GPUBuffer
    numParticlesBuffer: GPUBuffer
    densityBuffer: GPUBuffer
    mouseInfoUniformBuffer: GPUBuffer
    sphereRadiusBuffer: GPUBuffer
    initBoxSizeBuffer: GPUBuffer
    numParticles = 0
    gridCount = 0
    maxGridCount = 0
    maxParticleCount = 0
    densityGridCount = 0

    clearGridPipeline: GPUComputePipeline
    clearDensityGridPipeline: GPUComputePipeline
    castDensityGridPipeline: GPUComputePipeline
    p2g1Pipeline: GPUComputePipeline
    p2g2Pipeline: GPUComputePipeline
    p2gDensityPipeline: GPUComputePipeline
    updateGridPipeline: GPUComputePipeline
    g2pPipeline: GPUComputePipeline
    copyPositionPipeline: GPUComputePipeline

    clearGridBindGroup: GPUBindGroup
    clearDensityGridBindGroup: GPUBindGroup
    castDensityGridBindGroup: GPUBindGroup
    p2g1BindGroup: GPUBindGroup
    p2g2BindGroup: GPUBindGroup
    p2gDensityBindGroup: GPUBindGroup
    updateGridBindGroup: GPUBindGroup
    g2pBindGroup: GPUBindGroup
    copyPositionBindGroup: GPUBindGroup

    particleBuffer: GPUBuffer
    dtBuffer: GPUBuffer
    densityGridBuffer: GPUBuffer

    device: GPUDevice

    renderDiameter: number

    frameCount: number

    spawned: boolean

    mouseInfoValues = new ArrayBuffer(32)
    mouseInfoViews = {
        screenSize: new Float32Array(this.mouseInfoValues, 0, 2),
        mouseCoord: new Float32Array(this.mouseInfoValues, 8, 2),
        mouseVel: new Float32Array(this.mouseInfoValues, 16, 2),
        mouseRadius: new Float32Array(this.mouseInfoValues, 24, 1),
    };

    restDensity: number

    constructor (
                particleBuffer: GPUBuffer, posvelBuffer: GPUBuffer, renderUniformBuffer: GPUBuffer,
                densityGridBuffer: GPUBuffer, castedDensityGridBuffer: GPUBuffer, initBoxSizeBuffer: GPUBuffer, densityGridSizeBuffer: GPUBuffer,
                device: GPUDevice, depthMapTextureView: GPUTextureView, canvas: HTMLCanvasElement,
                maxGridCount: number, maxParticleCount: number, fixedPointMultiplier: number, renderDiameter: number,
        )
    {
        this.device = device
        this.renderDiameter = renderDiameter
        this.frameCount = 0
        this.spawned = false
        this.numParticles = 0
        this.maxGridCount = maxGridCount
        this.maxParticleCount = maxParticleCount
        this.initBoxSizeBuffer = initBoxSizeBuffer
        this.initParticles = true;

        const commonWgsl = /*wgsl*/`
const SCALE : f32 = 65536.0; // 2^16 fixed point
const SCALE_INV : f32 = 1.0 / SCALE;
fn encodeFixedPoint(v: f32) -> i32 {
    return i32(v * SCALE + 0.5);
}
fn decodeFixedPoint(i: i32) -> f32 {
    return f32(i) * SCALE_INV;
}

const dynamicViscosity : f32 = 0.1;
const restDensity : f32 = 3.0;
const stiffness : f32 = 50.0;

struct Particle {
    position: vec3f,
    pad1: f32,
    v: vec3f,
    pad2: f32,
    C: mat3x3f,
}
struct Cell {
    vx: i32,
    vy: i32,
    vz: i32,
    mass: i32,
}

struct AtomicCell {
    vx: atomic<i32>,
    vy: atomic<i32>,
    vz: atomic<i32>,
    mass: atomic<i32>,
}

struct AtomicCellWithoutMass {
    vx: atomic<i32>,
    vy: atomic<i32>,
    vz: atomic<i32>,
    mass: i32,
}

struct PosVel {
    position: vec3f,
    v: vec3f,
}
        `

        const templateCode = (code) => { return  commonWgsl + code; }
        const createMod = (code) => device.createShaderModule({ code: templateCode(code) });

        const clearGridModule = createMod(/*wgsl*/`
@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;

@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cells)) {
        cells[id.x].mass = 0;
        cells[id.x].vx = 0;
        cells[id.x].vy = 0;
        cells[id.x].vz = 0;
    }
}
        `);
        const p2g1Module = createMod(/*wgsl*/`
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<AtomicCell>;
@group(0) @binding(2) var<uniform> initBoxSize: vec3f;
@group(0) @binding(3) var<uniform> numParticles: u32;

@compute @workgroup_size(64)
fn p2g_1(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < numParticles) {
        var weights: array<vec3f, 3>;

        let particle = particles[id.x];
        let cellIndex: vec3f = floor(particle.position);
        let cellDiff: vec3f = particle.position - (cellIndex + 0.5f);
        weights[0] = 0.5f * (0.5f - cellDiff) * (0.5f - cellDiff);
        weights[1] = 0.75f - cellDiff * cellDiff;
        weights[2] = 0.5f * (0.5f + cellDiff) * (0.5f + cellDiff);

        let C: mat3x3f = particle.C;

        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cellX: vec3f = vec3f(
                            cellIndex.x + f32(gx) - 1.,
                            cellIndex.y + f32(gy) - 1.,
                            cellIndex.z + f32(gz) - 1.
                        );
                    let cellDist = (cellX + 0.5f) - particle.position;

                    let Q: vec3f = C * cellDist;

                    let massContrib: f32 = weight * 1.0; // assuming particle.mass = 1.0
                    let velContrib: vec3f = massContrib * (particle.v + Q);
                    let cellIndex1D: i32 =
                        i32(cellX.x) * i32(initBoxSize.y) * i32(initBoxSize.z) +
                        i32(cellX.y) * i32(initBoxSize.z) +
                        i32(cellX.z);
                    atomicAdd(&cells[cellIndex1D].mass, encodeFixedPoint(massContrib));
                    atomicAdd(&cells[cellIndex1D].vx, encodeFixedPoint(velContrib.x));
                    atomicAdd(&cells[cellIndex1D].vy, encodeFixedPoint(velContrib.y));
                    atomicAdd(&cells[cellIndex1D].vz, encodeFixedPoint(velContrib.z));
                }
            }
        }
    }
}
        `);
        const p2g2Module = createMod(/*wgsl*/`
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<AtomicCellWithoutMass>;
@group(0) @binding(2) var<uniform> initBoxSize: vec3f;
@group(0) @binding(3) var<uniform> numParticles: u32;
@group(0) @binding(4) var<uniform> dt: f32;

@compute @workgroup_size(64)
fn p2g_2(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < numParticles) {
        var weights: array<vec3f, 3>;

        let particle = particles[id.x];
        let cellIndex: vec3f = floor(particle.position);
        let cellDiff: vec3f = particle.position - (cellIndex + 0.5f);
        weights[0] = 0.5f * (0.5f - cellDiff) * (0.5f - cellDiff);
        weights[1] = 0.75f - cellDiff * cellDiff;
        weights[2] = 0.5f * (0.5f + cellDiff) * (0.5f + cellDiff);

        var density: f32 = 0.;
        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cellX: vec3f = vec3f(
                            cellIndex.x + f32(gx) - 1.,
                            cellIndex.y + f32(gy) - 1.,
                            cellIndex.z + f32(gz) - 1.
                        );
                    let cellIndex1D: i32 =
                        i32(cellX.x) * i32(initBoxSize.y) * i32(initBoxSize.z) +
                        i32(cellX.y) * i32(initBoxSize.z) +
                        i32(cellX.z);
                    density += decodeFixedPoint(cells[cellIndex1D].mass) * weight;
                }
            }
        }

        let volume: f32 = 1.0 / density; // particle.mass = 1.0;
        // densities[id.x] = density;

        let pressure: f32 = max(-0.0, stiffness * (pow(density / restDensity, 1.) - 1));

        var stress: mat3x3f = mat3x3f(-pressure, 0, 0, 0, -pressure, 0, 0, 0, -pressure);
        let dudv: mat3x3f = particle.C;
        let strain: mat3x3f = dudv + transpose(dudv);
        stress += dynamicViscosity * strain;

        let eq_16_term0 = -volume * 4 * stress * dt;

        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cellX: vec3f = vec3f(
                            cellIndex.x + f32(gx) - 1.,
                            cellIndex.y + f32(gy) - 1.,
                            cellIndex.z + f32(gz) - 1.
                        );
                    let cellDist = (cellX + 0.5f) - particle.position;
                    let cellIndex1D: i32 =
                        i32(cellX.x) * i32(initBoxSize.y) * i32(initBoxSize.z) +
                        i32(cellX.y) * i32(initBoxSize.z) +
                        i32(cellX.z);
                    let momentum: vec3f = eq_16_term0 * weight * cellDist;
                    atomicAdd(&cells[cellIndex1D].vx, encodeFixedPoint(momentum.x));
                    atomicAdd(&cells[cellIndex1D].vy, encodeFixedPoint(momentum.y));
                    atomicAdd(&cells[cellIndex1D].vz, encodeFixedPoint(momentum.z));
                }
            }
        }
    }
}
`);
        const updateGridModule = createMod(/*wgsl*/`
@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> realBoxSize: vec3f;
@group(0) @binding(2) var<uniform> initBoxSize: vec3f;
@group(0) @binding(3) var<uniform> dt: f32;

@compute @workgroup_size(64)
fn updateGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cells)) {
        var forceDir = vec3f(0.);
        let dt = dt;
        if (cells[id.x].mass > 0) {
            var floatV: vec3f = vec3f(
                decodeFixedPoint(cells[id.x].vx),
                decodeFixedPoint(cells[id.x].vy),
                decodeFixedPoint(cells[id.x].vz)
            );
            floatV /= decodeFixedPoint(cells[id.x].mass);
            let strength = 0.0;
            cells[id.x].vx = encodeFixedPoint(floatV.x + strength * forceDir.x);
            cells[id.x].vy = encodeFixedPoint(floatV.y + strength * forceDir.y - 0.40 * dt);
            cells[id.x].vz = encodeFixedPoint(floatV.z + strength * forceDir.z);
            var x: i32 = i32(id.x) / i32(initBoxSize.z) / i32(initBoxSize.y);
            var y: i32 = (i32(id.x) / i32(initBoxSize.z)) % i32(initBoxSize.y);
            var z: i32 = i32(id.x) % i32(initBoxSize.z);
            if (x < 2 || x > i32(ceil(realBoxSize.x) - 3)) { cells[id.x].vx = 0; }
            if (y < 2 || y > i32(ceil(realBoxSize.y) - 3)) { cells[id.x].vy = 0; }
            if (z < 2 || z > i32(ceil(realBoxSize.z) - 3)) { cells[id.x].vz = 0; }
        }
    }
}
        `);
        const g2pModule = createMod(/*wgsl*/`
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> realBoxSize: vec3f;
@group(0) @binding(3) var<uniform> initBoxSize: vec3f;
@group(0) @binding(4) var<uniform> numParticles: u32;
@group(0) @binding(5) var<uniform> dt: f32;

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < numParticles) {
        particles[id.x].v = vec3f(0.);
        var weights: array<vec3f, 3>;

        let particle = particles[id.x];
        let cellIndex: vec3f = floor(particle.position);
        let cellDiff: vec3f = particle.position - (cellIndex + 0.5f);
        weights[0] = 0.5f * (0.5f - cellDiff) * (0.5f - cellDiff);
        weights[1] = 0.75f - cellDiff * cellDiff;
        weights[2] = 0.5f * (0.5f + cellDiff) * (0.5f + cellDiff);

        var B: mat3x3f = mat3x3f(vec3f(0.), vec3f(0.), vec3f(0.));
        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cellX: vec3f = vec3f(
                        cellIndex.x + f32(gx) - 1.,
                        cellIndex.y + f32(gy) - 1.,
                        cellIndex.z + f32(gz) - 1.
                    );
                    let cellDist: vec3f = (cellX + 0.5f) - particle.position;
                    let cellIndex1D: i32 =
                        i32(cellX.x) * i32(initBoxSize.y) * i32(initBoxSize.z) +
                        i32(cellX.y) * i32(initBoxSize.z) +
                        i32(cellX.z);
                    let weighted_velocity: vec3f = vec3f(
                        decodeFixedPoint(cells[cellIndex1D].vx),
                        decodeFixedPoint(cells[cellIndex1D].vy),
                        decodeFixedPoint(cells[cellIndex1D].vz)
                    ) * weight;
                    let term: mat3x3f = mat3x3f(
                        weighted_velocity * cellDist.x,
                        weighted_velocity * cellDist.y,
                        weighted_velocity * cellDist.z
                    );

                    B += term;

                    particles[id.x].v += weighted_velocity;
                }
            }
        }

        particles[id.x].C = B * 4.0f;
        particles[id.x].position += particles[id.x].v * dt;
        particles[id.x].position = vec3f(
            clamp(particles[id.x].position.x, 1., realBoxSize.x - 2.),
            clamp(particles[id.x].position.y, 1., realBoxSize.y - 2.),
            clamp(particles[id.x].position.z, 1., realBoxSize.z - 2.)
        );

        let center = vec3f(realBoxSize.x / 2, realBoxSize.y / 2, realBoxSize.z / 2);
        let dist = center - particles[id.x].position;
        let dirToOrigin = normalize(dist);
        var rForce = vec3f(0);


        let k = 2.0;
        let wallStiffness = 1.0;
        let x_n: vec3f = particles[id.x].position + particles[id.x].v * dt * k;
        let wallMin: vec3f = vec3f(3.);
        let wallMax: vec3f = realBoxSize - 4.;
        if (x_n.x < wallMin.x) { particles[id.x].v.x += wallStiffness * (wallMin.x - x_n.x); }
        if (x_n.x > wallMax.x) { particles[id.x].v.x += wallStiffness * (wallMax.x - x_n.x); }
        if (x_n.y < wallMin.y) { particles[id.x].v.y += wallStiffness * (wallMin.y - x_n.y); }
        if (x_n.y > wallMax.y) { particles[id.x].v.y += wallStiffness * (wallMax.y - x_n.y); }
        if (x_n.z < wallMin.z) { particles[id.x].v.z += wallStiffness * (wallMin.z - x_n.z); }
        if (x_n.z > wallMax.z) { particles[id.x].v.z += wallStiffness * (wallMax.z - x_n.z); }
    }
}
        `);
        const copyPositionModule = createMod(/*wgsl*/`
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> posvel: array<PosVel>;
@group(0) @binding(2) var<uniform> numParticles: u32;

@compute @workgroup_size(64)
fn copyPosition(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < numParticles) {
        posvel[id.x].position = particles[id.x].position;
        posvel[id.x].v = particles[id.x].v;
    }
}

        `);
        const resetParticlesModule = createMod(/*wgsl*/`
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> realBoxSize: vec3f;
@group(0) @binding(2) var<uniform> numParticles: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < numParticles) {
        var side = u32(pow(f32(numParticles), 1.0/3.0));
        let x = id.x % side;
        let y = id.x / side % side;
        let z = id.x / (side * side);
        var unitCube = vec3f(f32(x), f32(y), f32(z)) / f32(side);
        let pos = realBoxSize * .25 + unitCube * realBoxSize * .5;
        particles[id.x].position = pos;
    }
}
        `);

        this.clearGridPipeline = device.createComputePipeline({
            label: "clear grid pipeline",
            layout: 'auto',
            compute: {
                module: clearGridModule,
            }
        })
        this.p2g1Pipeline = device.createComputePipeline({
            label: "p2g 1 pipeline",
            layout: 'auto',
            compute: {
                module: p2g1Module,
            }
        })
        this.p2g2Pipeline = device.createComputePipeline({
            label: "p2g 2 pipeline",
            layout: 'auto',
            compute: {
                module: p2g2Module,
            }
        })
        this.updateGridPipeline = device.createComputePipeline({
            label: "update grid pipeline",
            layout: 'auto',
            compute: {
                module: updateGridModule,
            }
        });
        this.g2pPipeline = device.createComputePipeline({
            label: "g2p pipeline",
            layout: 'auto',
            compute: {
                module: g2pModule,
            }
        });
        this.copyPositionPipeline = device.createComputePipeline({
            label: "copy position pipeline",
            layout: 'auto',
            compute: {
                module: copyPositionModule,
            }
        });

        this.resetParticlesPipeline = device.createComputePipeline({
            label: "resetParticlesModule",
            layout: 'auto',
            compute: {
                module: resetParticlesModule,
            }
        });

        const cellBuffer = device.createBuffer({
            label: 'cells buffer',
            size: this.cellStructSize * maxGridCount,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.densityBuffer = device.createBuffer({
            label: 'density buffer',
            size: 4 * maxParticleCount,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.realBoxSizeBuffer = device.createBuffer({
            label: 'real box size buffer',
            size: 12, // 3 x f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.numParticlesBuffer = device.createBuffer({
            label: 'number of particles buffer',
            size: 4, // 1 x f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.mouseInfoUniformBuffer = device.createBuffer({
            label: 'mouse info buffer',
            size: this.mouseInfoValues.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.sphereRadiusBuffer = device.createBuffer({
            label: 'sphere radius buffer',
            size: 4, // 1 x f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.dtBuffer = device.createBuffer({
            label: 'dt buffer',
            size: 4, // 1 x f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        this.mouseInfoViews.screenSize.set([canvas.width, canvas.height]);
        this.device.queue.writeBuffer(this.mouseInfoUniformBuffer, 0, this.mouseInfoValues);

        // BindGroup
        this.clearGridBindGroup = device.createBindGroup({
            layout: this.clearGridPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: cellBuffer }},
            ],
        })
        this.p2g1BindGroup = device.createBindGroup({
            layout: this.p2g1Pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: cellBuffer }},
                { binding: 2, resource: { buffer: initBoxSizeBuffer }},
                { binding: 3, resource: { buffer: this.numParticlesBuffer }},
            ],
        })
        this.p2g2BindGroup = device.createBindGroup({
            layout: this.p2g2Pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: cellBuffer }},
                { binding: 2, resource: { buffer: initBoxSizeBuffer }},
                { binding: 3, resource: { buffer: this.numParticlesBuffer }},
                { binding: 4, resource: { buffer: this.dtBuffer }},
            ]
        })
        this.updateGridBindGroup = device.createBindGroup({
            layout: this.updateGridPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cellBuffer }},
                { binding: 1, resource: { buffer: this.realBoxSizeBuffer }},
                { binding: 2, resource: { buffer: initBoxSizeBuffer }},
                { binding: 3, resource: { buffer: this.dtBuffer }},
            ],
        })
        this.g2pBindGroup = device.createBindGroup({
            layout: this.g2pPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: cellBuffer }},
                { binding: 2, resource: { buffer: this.realBoxSizeBuffer }},
                { binding: 3, resource: { buffer: initBoxSizeBuffer }},
                { binding: 4, resource: { buffer: this.numParticlesBuffer }},
                { binding: 5, resource: { buffer: this.dtBuffer }},
            ],
        })
        this.copyPositionBindGroup = device.createBindGroup({
            layout: this.copyPositionPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: posvelBuffer }},
                { binding: 2, resource: { buffer: this.numParticlesBuffer }},
            ]
        })

        this.resetParticlesBindGroup = device.createBindGroup({
            layout: this.resetParticlesPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: this.initBoxSizeBuffer }},
                { binding: 2, resource: { buffer: this.numParticlesBuffer }},
            ]
        })


        this.particleBuffer = particleBuffer
        this.densityGridBuffer = densityGridBuffer
    }

    initDambreak(initBoxSize: number[], numParticles: number) {
        let particlesBuf = new ArrayBuffer(mlsmpmParticleStructSize * this.maxParticleCount);
        const spacing = 0.9 ;

        this.numParticles = numParticles;

        let sphereCenter = [initBoxSize[0] / 2, initBoxSize[0] / 2, initBoxSize[2] / 2]

        console.log(initBoxSize);

        // for (let j = 3; j < initBoxSize[1] * 0.80 && this.numParticles < numParticles; j += spacing) {
        //     for (let i = initBoxSize[0] * 0.25; i < initBoxSize[0] - 4 && this.numParticles < numParticles; i += spacing) {
        //         for (let k = 3; k < initBoxSize[2] / 2 && this.numParticles < numParticles; k += spacing) {
        //             const offset = mlsmpmParticleStructSize * this.numParticles;
        //             const particleViews = {
        //                 position: new Float32Array(particlesBuf, offset + 0, 3),
        //                 v: new Float32Array(particlesBuf, offset + 16, 3),
        //                 C: new Float32Array(particlesBuf, offset + 32, 12),
        //             };
        //             const jitter = 0.5 * Math.random();
        //             particleViews.position.set([i + jitter, j + jitter, k + jitter]);
        //             // console.log([i + jitter, j + jitter, k + jitter]);
        //             this.numParticles++;
        //         }
        //     }
        // }

        // console.log(this.numParticles)
        // if (this.numParticles < numParticles) {
        //     console.log("warning: actual number of particles is smaller than the specified number. make bounding box larger.")
        // }

        let particles = new ArrayBuffer(mlsmpmParticleStructSize * this.numParticles);
        const oldView = new Uint8Array(particlesBuf);
        const newView = new Uint8Array(particles);
        newView.set(oldView.subarray(0, newView.length));

        return particles;
    }

    reset(initBoxSize: number[], numParticles: number) {
        this.gridCount = Math.ceil(initBoxSize[0]) * Math.ceil(initBoxSize[1]) * Math.ceil(initBoxSize[2])
        if (this.gridCount > this.maxGridCount) {
            throw new Error("gridCount should be equal to or less than maxGridCount")
        }
        this.densityGridCount = this.gridCount
        const initBoxSizeArray = new Float32Array(initBoxSize)
        this.device.queue.writeBuffer(this.initBoxSizeBuffer, 0, initBoxSizeArray)
        this.frameCount = 0;
        let particles = this.initDambreak(initBoxSize, numParticles)
        this.device.queue.writeBuffer(this.particleBuffer, 0, particles)
        this.changeBoxSize(initBoxSize)
        this.changeNumParticles(this.numParticles)
        this.initParticles = true;
    }

    execute(commandEncoder: GPUCommandEncoder, mouseCoord: number[], mouseVel: number[], mouseRadius: number,
        densityGridFlag: boolean, dt: number, running: boolean, densityGridSize: number[]
    ) {
        const computePass = commandEncoder.beginComputePass();

        this.mouseInfoViews.mouseCoord.set([mouseCoord[0], mouseCoord[1]])
        this.mouseInfoViews.mouseVel.set([mouseVel[0], mouseVel[1]])
        this.mouseInfoViews.mouseRadius.set([mouseRadius])
        this.device.queue.writeBuffer(this.mouseInfoUniformBuffer, 0, this.mouseInfoValues);

        // console.log(dt);
        const dtArray = new Float32Array([dt])
        this.device.queue.writeBuffer(this.dtBuffer, 0, dtArray)

        if (running) {
            // for (let i = 0; i < 1; i++) {  // single timestep!!!
                if (this.initParticles) {
                    computePass.setBindGroup(0, this.resetParticlesBindGroup)
                    computePass.setPipeline(this.resetParticlesPipeline)
                    computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
                    this.initParticles = false;
                }

                computePass.setBindGroup(0, this.clearGridBindGroup);
                computePass.setPipeline(this.clearGridPipeline);
                computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 64))
                computePass.setBindGroup(0, this.p2g1BindGroup)
                computePass.setPipeline(this.p2g1Pipeline)
                computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
                computePass.setBindGroup(0, this.p2g2BindGroup)
                computePass.setPipeline(this.p2g2Pipeline)
                computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
                computePass.setBindGroup(0, this.updateGridBindGroup)
                computePass.setPipeline(this.updateGridPipeline)
                computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 64))
                computePass.setBindGroup(0, this.g2pBindGroup)
                computePass.setPipeline(this.g2pPipeline)
                computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
            // }
            computePass.setBindGroup(0, this.copyPositionBindGroup)
            computePass.setPipeline(this.copyPositionPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
        }

        computePass.end()

        this.frameCount++;
    }

    changeBoxSize(realBoxSize: number[]) {
        const realBoxSizeArray = new Float32Array(realBoxSize);
        this.device.queue.writeBuffer(this.realBoxSizeBuffer, 0, realBoxSizeArray)
    }

    changeNumParticles(numParticles: number) {
        const numParticlesArray = new Int32Array([numParticles])
        this.device.queue.writeBuffer(this.numParticlesBuffer, 0, numParticlesArray)
        this.numParticles = numParticles
    }
}
