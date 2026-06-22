use std::sync::LazyLock;

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};
use noise::NoiseFn;
use num::ToPrimitive;

use crate::{coordinates::bary_iterative::BaryIterative, math::almost_equal};

/// Number of times a triangle mesh is subdivided
pub const MESH_SUBDIVISIONS: u32 = 2;
/// Number of triangles along one edge of the mesh
pub const MESH_STEPS: u32 = 1 << MESH_SUBDIVISIONS;

static BASH_MESH: LazyLock<Mesh> = LazyLock::new(|| create_bary_mesh(MESH_SUBDIVISIONS));

/// Create a base mesh where vertices are barycentric weights
fn create_bary_mesh(subdivisions: u32) -> Mesh {
    const ODD_OFFSETS: [UVec3; 3] = [UVec3::X, UVec3::Y, UVec3::Z];
    const EVEN_OFFSETS: [UVec3; 3] = [
        UVec3::from_array([0, 1, 1]),
        UVec3::from_array([1, 0, 1]),
        UVec3::from_array([1, 1, 0]),
    ];

    let n = 2_u32.pow(subdivisions);
    let num_vertices = (n + 1) * (n + 2) / 2;

    let mut indices = Vec::with_capacity(num_vertices as usize * 3);

    // Odds
    let n_odd = n;
    for x in 0..n_odd {
        for y in 0..(n_odd - x) {
            let z = n_odd - x - y - 1;
            let base = UVec3::new(x, y, z);

            let vertex_indices =
                ODD_OFFSETS.map(|o| BaryIterative::new(base + o, 1.).to_mesh_index());

            indices.extend(vertex_indices);
        }
    }

    // Evens
    let n_even = n_odd - 1;
    for x in 0..n_even {
        for y in 0..(n_even - x) {
            let z = n_even - x - y - 1;
            let base = UVec3::new(x, y, z);

            let vertex_indices =
                EVEN_OFFSETS.map(|o| BaryIterative::new(base + o, 1.).to_mesh_index());

            indices.extend(vertex_indices);
        }
    }

    // Vertices
    // let mut vertices = Vec::with_capacity(num_vertices as usize);
    let mut vertices = vec![Vec3A::ZERO; num_vertices as usize];
    for x in 0..(n + 1) {
        for y in 0..(n + 1 - x) {
            let z = n - x - y;

            let bary = BaryIterative::new(UVec3::new(x, y, z), 1.);
            let vertex = Vec3A::from_array(bary.as_ratios().map(|r| r.to_f32().unwrap()));
            let index = bary.to_mesh_index();
            vertices[index as usize] = vertex;
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_indices(Indices::U32(indices))
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
}

/// Takes a mesh on the unit sphere and applies the noise function.
pub fn apply_noise_to_mesh(mut mesh: Mesh, noise_gen: impl NoiseFn<f64, 3>) -> Mesh {
    let positions = mesh
        .attribute_mut(Mesh::ATTRIBUTE_POSITION)
        .expect("base mesh has vertices");
    let VertexAttributeValues::Float32x3(positions) = positions else {
        unreachable!("Bad vertex type");
    };

    // Walk over sphere points and apply noise map
    for p in positions {
        assert!(almost_equal(Vec3A::from_array(*p).length_squared(), 1.));
        let height = noise_gen.get(p.map(|v| v as f64)) as f32;
        let height = (1. + height / 2.).clamp(0.1, f32::MAX);

        p.iter_mut().for_each(|v| *v *= height);
    }

    mesh.with_computed_normals()
}

/// Creates a full mesh with all the bells and whistles
pub fn create_mesh(triangle: [Vec3A; 3], noise_gen: impl NoiseFn<f64, 3>) -> Mesh {
    let mut mesh = BASH_MESH.clone();

    let positions = mesh
        .attribute_mut(Mesh::ATTRIBUTE_POSITION)
        .expect("base mesh has vertices");
    let VertexAttributeValues::Float32x3(positions) = positions else {
        unreachable!("Bad vertex type");
    };

    for p in positions {
        // Convert barycentric coords to cartesian
        let cart = BaryIterative::from_float_weights(*p, 1., MESH_SUBDIVISIONS)
            .to_cartesian(triangle)
            .to_array();

        // Apply noise function
        let height = noise_gen.get(cart.map(|v| v as f64)) as f32;
        let height = (1. + height / 2.).clamp(0.1, f32::MAX);

        *p = cart.map(|v| v * height);
    }

    for t in mesh.triangles().unwrap() {
        let normal = t.normal().unwrap();
        assert!(normal.dot(t.vertices[0]) > 0., "Normal not facing outwards");
    }

    mesh.with_computed_normals()
}
