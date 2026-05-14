use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use num::ToPrimitive;

use crate::bary::BarycentricSnapped;

/// Index into the mesh vertices
fn bary_to_index(bary: BarycentricSnapped) -> u32 {
    let [x, y, z] = bary.distances.to_array();
    assert!(x + y + z == bary.denominator, "{:?}", bary);

    let a = bary.denominator + 1;
    let offset = ((2 * a + 1 - z) * z) / 2;
    offset + x
}

fn create_bary_mesh(subdivisions: u32) -> Mesh {
    const ODD_OFFSETS: [UVec3; 3] = [UVec3::X, UVec3::Y, UVec3::Z];
    const EVEN_OFFSETS: [UVec3; 3] = [
        UVec3::from_array([0, 1, 1]),
        UVec3::from_array([1, 0, 1]),
        UVec3::from_array([1, 1, 0]),
    ];

    let n = 2_u32.pow(subdivisions);
    let num_vertices = n * (n + 1) / 2;

    let mut indices = Vec::with_capacity(num_vertices as usize * 3);

    // Odds
    let n_odd = n;
    for x in 0..n_odd {
        for y in 0..(n_odd - x) {
            let z = n_odd - x - y - 1;
            let base = UVec3::new(x, y, z);

            let vertex_indices = ODD_OFFSETS
                .map(|o| BarycentricSnapped::new(base + o, 1.))
                .map(bary_to_index);

            indices.extend(vertex_indices);
        }
    }

    // Evens
    let n_even = n_odd - 1;
    for x in 0..n_even {
        for y in 0..(n_even - x) {
            let z = n_even - x - y - 1;
            let base = UVec3::new(x, y, z);

            let vertex_indices = EVEN_OFFSETS
                .map(|o| BarycentricSnapped::new(base + o, 1.))
                .map(bary_to_index);

            indices.extend(vertex_indices);
        }
    }

    // Vertices
    let mut vertices = Vec::with_capacity(num_vertices as usize);
    for x in 0..(n + 1) {
        for y in 0..(n + 1 - x) {
            let z = n - x - y;

            let bary = BarycentricSnapped::new(UVec3::new(x, y, z), 1.);
            let vertex = Vec3A::from_array(bary.as_ratios().map(|r| r.to_f32().unwrap()));
            vertices.push(vertex);
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

#[cfg(test)]
mod tests {
    use crate::mesh::create_bary_mesh;

    #[test]
    fn test_mesh() {
        let mesh = create_bary_mesh(2);
        println!("mesh {:?}", mesh);
        panic!()
    }
}
