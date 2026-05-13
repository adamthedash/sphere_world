use bevy::prelude::*;

// ========================================================
// Trig & maths
// ========================================================

const EPS: f32 = 1e-3;

/// https://en.wikipedia.org/wiki/Triple_product#Properties
fn coplanar(v0: Vec3A, v1: Vec3A, point: Vec3A) -> bool {
    v0.cross(v1).dot(point).abs() < EPS
}

fn colinear(v: Vec3A, point: Vec3A) -> bool {
    (v.dot(point) - v.length() * point.length()).abs() < EPS
}

fn obtuse(v: Vec3A, point: Vec3A) -> bool {
    v.dot(point) < 0.
}

/// Project point towards the origin onto the line formed by v0-v1.
/// v0, v1, point must be coplanar
fn project(v0: Vec3A, v1: Vec3A, point: Vec3A) -> Vec3A {
    let d = v1 - v0; // direction of the line

    // Solve: s * P = v0 + t*d  =>  s * P - t*d = v0
    // Use cross products to isolate t:
    let p_cross_d = point.cross(d);
    let denom = p_cross_d.dot(p_cross_d); // |P × d|²

    let v0_cross_d = v0.cross(d);
    let s = v0_cross_d.dot(p_cross_d) / denom;

    s * point
}

/// Cosine similarity: (a . b) / |a||b|
fn cossim(v0: Vec3A, v1: Vec3A) -> f32 {
    v0.dot(v1) / (v0.length_squared() * v1.length_squared()).sqrt()
}

/// Absolute angle between two vectors
pub fn arc_distance(v0: Vec3A, v1: Vec3A) -> f32 {
    cossim(v0, v1).acos()
}

/// https://en.wikipedia.org/wiki/Heron%27s_formula
fn heron_area(a: f32, b: f32, c: f32) -> f32 {
    let s = (a + b + c) / 2.;

    (s * (s - a) * (s - b) * (s - c)).sqrt()
}

pub fn almost_equal(a: f32, b: f32) -> bool {
    (a - b).abs() < EPS
}
