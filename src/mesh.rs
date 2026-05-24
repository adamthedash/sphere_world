use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};
use num::ToPrimitive;

use crate::coordinates::bary_iterative::BaryIterative;

/// Number of times a triangle mesh is subdivided
pub const MESH_SUBDIVISIONS: u32 = 3;
/// Number of triangles along one edge of the mesh
pub const MESH_STEPS: u32 = 1 << MESH_SUBDIVISIONS;

pub fn create_bary_mesh(subdivisions: u32) -> Mesh {
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
    .with_computed_normals()
}

/// Wrap a base mesh onto the surface of a sphere bounded by the provided triangle
pub fn base_mesh_to_triangle(mut mesh: Mesh, triangle: [Vec3A; 3]) -> Mesh {
    let positions = mesh
        .attribute_mut(Mesh::ATTRIBUTE_POSITION)
        .expect("base mesh has vertices");
    let VertexAttributeValues::Float32x3(positions) = positions else {
        unreachable!("Bad vertex type");
    };

    // Walk over bary and convert to cartesian
    for p in positions {
        let b = BaryIterative::from_float_weights(*p, 1., MESH_SUBDIVISIONS);
        *p = BaryIterative::from_float_weights(*p, 1., MESH_SUBDIVISIONS)
            .to_cartesian(triangle)
            .to_array();
    }

    // TODO: Normals seem to be inverted for meshes with high subdivisions? i.e. they are dark on
    // the lit side of the world and vice versa
    mesh.with_computed_normals()
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{FRAC_PI_4, PI};

    use bevy::mesh::{Mesh, VertexAttributeValues};
    use glam::Vec3A;

    use crate::{
        mesh::{base_mesh_to_triangle, create_bary_mesh},
        triangle::Triangle,
    };

    #[test]
    fn test_mesh() {
        fn get_vertices(mesh: &Mesh) -> &[[f32; 3]] {
            let positions = mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .expect("base mesh has vertices");
            let VertexAttributeValues::Float32x3(positions) = positions else {
                unreachable!("Bad vertex type");
            };

            positions
        }

        fn print_mesh(mesh: &Mesh) {
            let positions = get_vertices(mesh);

            for (i, p) in positions.iter().enumerate() {
                println!("{i} {p:.2?}");
            }

            let indices = mesh.indices().unwrap();
            indices.iter().array_chunks::<3>().for_each(|triangle| {
                println!("{triangle:?}");
            });
        }

        let mesh = create_bary_mesh(1);
        print_mesh(&mesh);
        println!();
        assert_eq!(
            get_vertices(&mesh),
            &[
                [0., 1., 0.],
                [0.5, 0.5, 0.],
                [1., 0., 0.,],
                [0., 0.5, 0.5],
                [0.5, 0., 0.5],
                [0., 0., 1.],
            ]
        );

        let triangle = Triangle::new([Vec3A::X, Vec3A::Y, Vec3A::Z]);
        let mesh = base_mesh_to_triangle(mesh, triangle.vertices);
        print_mesh(&mesh);
        let x = FRAC_PI_4.sin();
        assert_eq!(
            get_vertices(&mesh),
            &[
                [0., 1., 0.],
                [x, x, 0.],
                [1., 0., 0.,],
                [0., x, x],
                [x, 0., x],
                [0., 0., 1.],
            ]
        );
    }
}
