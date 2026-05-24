use std::f32::consts::TAU;

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use itertools::Itertools;
use num::ToPrimitive;

use crate::{coordinates::bary_iterative::BaryIterative, math::arc_distance};

#[derive(Debug, Clone, Copy)]
pub enum TrianglePointCmp {
    Outside,
    /// Index of triangle vertex
    Corner(usize),
    Edge {
        v0: usize,
        v1: usize,
        /// How far along V0-V1
        t: f32,
    },
    Inside,
}

#[derive(Debug, Clone, Copy)]
pub enum TriangleTriangleCmp {
    Same,
    Unrelated,
    Sibling,
    Unc,
    Ancestor,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct Triangle {
    pub vertices: [Vec3A; 3],
    pub centre: Vec3A,
    pub normal: Vec3A,
    /// 0-1-2-0
    pub edges: [(Vec3A, Vec3A); 3],
    pub edge_midpoints: [Vec3A; 3],
}

impl Triangle {
    pub fn new(vertices: [Vec3A; 3]) -> Self {
        let centre = vertices.iter().sum::<Vec3A>() / 3.;

        let edges = vertices
            .iter()
            .copied()
            .circular_tuple_windows::<(_, _)>()
            .collect_array::<3>()
            .unwrap();

        let normal = (edges[0].1 - edges[0].0)
            .cross(edges[1].1 - edges[1].0)
            .normalize();

        let edge_midpoints = edges.map(|(v0, v1)| v0.midpoint(v1));

        Self {
            vertices,
            centre,
            normal,
            edges,
            edge_midpoints,
        }
    }

    /// Distance between centre point and edge midpoint in radians
    pub fn edge_arc_radius(&self) -> f32 {
        let edge_midpoint = self.edge_midpoints[0];

        arc_distance(self.centre, edge_midpoint)
    }

    /// Get the bary corredinate of this point if it lies within the triangle
    pub fn cmp_point_bary(&self, point: Vec3A, subdivisions: u32) -> Option<BaryIterative> {
        BaryIterative::from_cartesian(self.vertices, point, subdivisions)
    }

    pub fn cmp_point(&self, point: Vec3A, subdivisions: u32) -> TrianglePointCmp {
        let Some(bary) = self.cmp_point_bary(point, subdivisions) else {
            return TrianglePointCmp::Outside;
        };

        if let Some(i) = bary.corner() {
            return TrianglePointCmp::Corner(i);
        }

        if let Some((v0, v1, weight)) = bary.edge() {
            return TrianglePointCmp::Edge {
                v0,
                v1,
                t: weight.to_f32().unwrap(),
            };
        }

        // Inside
        TrianglePointCmp::Inside
    }

    pub fn cmp_triangle(&self, other: &Self, subdivisions: u32) -> TriangleTriangleCmp {
        let cmp = self.vertices.map(|v| other.cmp_point(v, subdivisions));

        let mut cmp_counts = [0_usize; 4];
        for c in cmp {
            let i = match c {
                TrianglePointCmp::Outside => 0,
                TrianglePointCmp::Corner(_) => 1,
                TrianglePointCmp::Edge { .. } => 2,
                TrianglePointCmp::Inside => 3,
            };
            cmp_counts[i] += 1;
        }

        match cmp_counts {
            // [O C E I]
            // Ancestor fully inside, inside touching edge
            [_, _, _, 1..] |
            // Parent centre
            [_, _, 3.., _] |
            // Ancestor corner
            [_, 1, 2, _] => {
                TriangleTriangleCmp::Ancestor
            }

            // (gr)unc corner-edge
            [1, 1, 1, _] |
            // (gr)unc edge-edge
            [1, _, 2, _] |
            // Corner on edge but not adjacent
            [2, _, 1, _] => {
                TriangleTriangleCmp::Unc
            }

            // Direct sibling
            [1, 2, _, _] => {
                TriangleTriangleCmp::Sibling
            }
            // Self
            [_, 3.., _, _] => {
                TriangleTriangleCmp::Same
            }
            // Fully outside
            [3.., _, _, _] |
            // Shared corner but not adjacent
            [2, 1, _, _] => {
                TriangleTriangleCmp::Unrelated
            }
            x => {
                let cmp_pretty = "OCEI".chars().zip(x)
                    .filter(|(_, n)| *n > 0)
                    .map(|(c, n)| format!("{c}{n} "))
                    .collect::<String>();
                unreachable!("Invalid combination: {:?} {:?} -> {:?}", cmp_pretty, self.vertices, other.vertices);
            },
        }
    }

    pub fn subdivide(&self) -> [Self; 4] {
        // Create midpoints of edges on unit sphere
        let midpoints = self.edge_midpoints.map(Vec3A::normalize);

        // Create outer triangles
        let outer = midpoints
            .iter()
            .circular_tuple_windows()
            .zip(self.vertices.iter().cycle().skip(1))
            // CCW order
            .map(|((v0, v1), v2)| [*v2, *v1, *v0]);

        // Inner triangle is just between the new midpoints
        let inner = std::iter::once(midpoints);

        // Wrap in triangle
        outer
            .chain(inner)
            .map(Triangle::new)
            .collect_array()
            .unwrap()
    }

    // Gets a basic mesh of a single triangle for this chunk
    pub fn get_mesh(&self) -> Mesh {
        let indices = Indices::U32(vec![0, 1, 2]);
        let positions = self
            .vertices
            .iter()
            .map(|t| t.to_array())
            .collect::<Vec<_>>();

        // Set UV to point at the orientation of the triangle
        let uvs = (0..3)
            .map(|i| {
                let angle = TAU * i as f32 / 3.;
                let (y, x) = angle.sin_cos();
                [y, x].map(|x| x / 2. + 0.5)
            })
            .collect::<Vec<_>>();

        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_indices(indices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_computed_normals()
    }
}

#[cfg(test)]
mod tests {
    use std::{assert_matches, f32::consts::GOLDEN_RATIO};

    use glam::Vec3A;

    use crate::{
        coordinates::bary_iterative::BaryIterative,
        triangle::{Triangle, TrianglePointCmp},
    };

    #[test]
    fn test_cmp_point() {
        let triangle = Triangle::new(
            [
                Vec3A::new(0., 1., GOLDEN_RATIO),
                Vec3A::new(1., GOLDEN_RATIO, 0.),
                Vec3A::new(GOLDEN_RATIO, 0., 1.),
            ]
            .map(|v| v.normalize()),
        );

        // Corner
        let point = triangle.vertices[0];
        let cmp = triangle.cmp_point_bary(point, 2).unwrap();
        assert_eq!(cmp, BaryIterative::new([4, 0, 0].into(), cmp.length));

        // Edge midpoint
        let point = triangle.edge_midpoints[0];
        let cmp = triangle.cmp_point_bary(point, 2).unwrap();
        assert_eq!(cmp, BaryIterative::new([2, 2, 0].into(), cmp.length));
    }

    #[test]
    fn test_cmp_triangle() {
        // Edge case - Should be O O C, got E O C
        let triangle0 = Triangle::new([
            Vec3A::new(-0.2763932, -0.4472136, 0.8506508),
            Vec3A::new(-0.68819094, -0.52573115, 0.5),
            Vec3A::new(-0.16245984, -0.8506508, 0.49999997), // shared
        ]);
        let triangle1 = Triangle::new([
            Vec3A::new(0.0, -1.0, 0.0),
            Vec3A::new(-0.16245984, -0.8506508, 0.49999997), // shared
            Vec3A::new(-0.5257311, -0.8506508, 0.0),
        ]);

        let cmp = triangle0.cmp_triangle(&triangle1, 1);
        // assert_matches!(cmp, TriangleTriangleCmp::Unrelated, "{cmp:?}");

        let cmps = triangle0.vertices.map(|v| triangle1.cmp_point(v, 1));
        assert_matches!(
            cmps,
            [
                TrianglePointCmp::Outside,
                TrianglePointCmp::Outside,
                TrianglePointCmp::Corner(_)
            ],
            "{cmp:?}"
        );
    }
}
