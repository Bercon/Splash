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
