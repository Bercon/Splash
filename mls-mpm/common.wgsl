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
