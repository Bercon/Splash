import clearGrid from './clearGrid.wgsl'
import p2g_1 from './p2g_1.wgsl'
import p2g_2 from './p2g_2.wgsl'
import updateGrid from './updateGrid.wgsl'
import g2p from './g2p.wgsl'
import copyPosition from './copyPosition.wgsl'
import commonWgsl from './common.wgsl'
import resetParticlesWgsl from './resetParticles.wgsl'

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

        const templateCode = (code) => { return  commonWgsl + code; }
        const createMod = (code) => device.createShaderModule({ code: templateCode(code) });

        const clearGridModule = createMod(clearGrid);
        const p2g1Module = createMod(p2g_1);
        const p2g2Module = createMod(p2g_2);
        const updateGridModule = createMod(updateGrid);
        const g2pModule = createMod(g2p);
        const copyPositionModule = createMod(copyPosition);
        const resetParticlesModule = createMod(resetParticlesWgsl);

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
