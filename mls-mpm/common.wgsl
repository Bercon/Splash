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
