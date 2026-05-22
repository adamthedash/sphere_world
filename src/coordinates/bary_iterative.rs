use std::{fmt::Display, ops::Sub};

use glam::{FloatExt, IVec3, UVec3, Vec3A};
use itertools::Itertools;
use num::{Integer, ToPrimitive, rational::Ratio};

use crate::math::{EPS, almost_equal};

#[derive(Debug, Clone, Copy)]
pub struct BaryIterative {
    /// Barycentric-style weights over the geodesic triangle grid
    pub weights: UVec3,
    /// Sum of weights
    pub denominator: u32,
    /// Vector length
    pub length: f32,
}

impl PartialEq for BaryIterative {
    fn eq(&self, other: &Self) -> bool {
        self.weights == other.weights && almost_equal(self.length, other.length)
    }
}

impl Display for BaryIterative {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} / {}", self.weights.to_array(), self.denominator)
    }
}

/// Difference
impl Sub for BaryIterative {
    type Output = IVec3;

    fn sub(self, rhs: Self) -> Self::Output {
        self.weights.as_ivec3() - rhs.weights.as_ivec3()
    }
}

#[derive(Debug)]
enum PlaneCmp {
    Neg,
    Zero,
    Pos,
}

impl BaryIterative {
    pub fn new(weights: UVec3, length: f32) -> Self {
        let denominator = weights.element_sum();
        Self {
            weights,
            denominator,
            length,
        }
    }

    /// Iteratively subdivides the triangle until it finds a point close to the provided point
    pub fn from_cartesian(
        triangle: [Vec3A; 3],
        point: Vec3A,
        max_subidivisions: u32,
    ) -> Option<Self> {
        for v in triangle {
            assert!(v.is_normalized(), "Traingle must be geocentric");
        }

        let is_ccw = triangle[0].cross(triangle[1]).dot(triangle[2]) > 0.;

        let (point, length) = point.normalize_and_length();

        // Start by checking if the point lies within the triangle
        let plane_normals = triangle
            .iter()
            .circular_tuple_windows()
            .map(|(v0, v1)| if is_ccw { v0.cross(*v1) } else { v1.cross(*v0) }.normalize())
            .collect_array::<3>()
            .unwrap();

        // +ve == left of plane i.e. towards inside
        // any 2 == 0 -> on corner
        // any 1 == 0 -> on edge
        // any < 0 -> outside
        // all > 0 -> inside
        let plane_distances = plane_normals.map(|n| n.dot(point));
        let plane_cmps = plane_distances.map(|d| {
            const NEG_EPS: f32 = -EPS;
            match d {
                ..NEG_EPS => PlaneCmp::Neg,
                NEG_EPS..EPS => PlaneCmp::Zero,
                EPS.. => PlaneCmp::Pos,
                _ => unreachable!(),
            }
        });

        let edge_weights = |i0: usize, i1: usize| {
            // Edge vi0 -> vi1
            let bot = triangle[i0].angle_between(triangle[i1]);
            let top = triangle[i0].angle_between(point);
            let wi1 = top / bot;

            // Round to nearest grid point
            let wi1 = (wi1 * (1 << max_subidivisions) as f32).round() as u32;
            let wi0 = (1 << max_subidivisions) - wi1;

            let mut weights = [0; 3];
            weights[i0] = wi0;
            weights[i1] = wi1;

            weights
        };

        let corner_weights = |i: usize| {
            let mut weights = [0; 3];
            weights[i] = 1 << max_subidivisions;
            weights
        };

        use PlaneCmp::*;
        let weights = match plane_cmps {
            [Pos, Pos, Pos] => {
                // Inside
                if max_subidivisions == 0 {
                    // We've recursed too far, round to the nearest grid point
                    let corner = triangle
                        .iter()
                        .map(|v| v.angle_between(point))
                        .position_min_by(f32::total_cmp)
                        .unwrap();

                    corner_weights(corner)
                } else {
                    // Subdivide and check sub-triangles

                    // Create new triangles
                    let t = triangle;
                    let midpoints = [
                        t[0].slerp(t[1], 0.5), //
                        t[1].slerp(t[2], 0.5),
                        t[2].slerp(t[0], 0.5),
                    ];

                    let new_triangles = [
                        // Outer
                        [t[0], midpoints[0], midpoints[2]],
                        [t[1], midpoints[1], midpoints[0]],
                        [t[2], midpoints[2], midpoints[1]],
                        // Inner
                        midpoints,
                    ];

                    // New triangles are defined relative to these points on the parent
                    let new_relative_u32 = [
                        // Outer
                        [[2, 0, 0], [1, 1, 0], [1, 0, 1]],
                        [[0, 2, 0], [0, 1, 1], [1, 1, 0]],
                        [[0, 0, 2], [1, 0, 1], [0, 1, 1]],
                        // Inner
                        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
                    ];

                    // See which sub-triangle the point lives in
                    // NOTE: If it lies on a shared edge/corner, the first will be returned. But either
                    // would be valid
                    let (child_weights, relative) = new_triangles
                        .into_iter()
                        .zip(new_relative_u32)
                        .flat_map(|(t, r)| {
                            let bary =
                                BaryIterative::from_cartesian(t, point, max_subidivisions - 1);
                            bary.map(|b| (b.weights, r))
                        })
                        .next()
                        .expect(
                            "point lies within this triangle, so it must return a valid answer",
                        );

                    // Convert the relative weights back to parent
                    let mut weights = [0; 3];
                    for (cw, r) in child_weights.to_array().into_iter().zip(relative) {
                        for (w, r) in weights.iter_mut().zip(r) {
                            *w += cw * r;
                        }
                    }

                    weights
                }
            }

            // Corner v0
            [Zero, Pos, Zero] => corner_weights(0),
            // Corner v1
            [Zero, Zero, Pos] => corner_weights(1),
            // Corner v2
            [Pos, Zero, Zero] => corner_weights(2),

            // Edge v0 -> v1
            [Zero, Pos, Pos] => edge_weights(0, 1),
            // Edge v1 -> v2
            [Pos, Zero, Pos] => edge_weights(1, 2),
            // Edge v2 -> v0
            [Pos, Pos, Zero] => edge_weights(2, 0),

            // Outside
            _ => return None,
        };

        Some(Self {
            weights: weights.into(),
            denominator: 1 << max_subidivisions,
            length,
        })
    }

    /// Create a bary from a set of float weights which sum to 1.  
    /// Weights must represent a grid point at the provided subdivision level
    pub fn from_float_weights(weights: [f32; 3], length: f32, subdivisions: u32) -> Self {
        assert!(
            almost_equal(weights.iter().sum::<f32>(), 1.),
            "Sum({:?}) != 1",
            weights
        );
        assert!(weights.iter().all(|w| *w >= 0.));

        let denominator = 1 << subdivisions;

        let weights = weights.map(|w| (w * denominator as f32).round() as u32);
        assert_eq!(
            weights.iter().sum::<u32>(),
            denominator,
            "lost precision on weight rounding"
        );

        Self {
            weights: weights.into(),
            denominator,
            length,
        }
    }

    pub fn to_cartesian(self, triangle: [Vec3A; 3]) -> Vec3A {
        for v in triangle {
            assert!(v.is_normalized(), "Traingle must be geocentric");
        }

        // Slerp along one edge
        let edge_weight_v0_v1 = if self.weights[1] == 0 {
            0.
        } else {
            self.weights[1] as f32 / (self.denominator - self.weights[2]) as f32
        };

        let edge_weight_v2_v1 = edge_weight_v0_v1;

        let mid_weight = if self.weights[2] == 0 {
            0.
        } else {
            self.weights[2] as f32 / (self.denominator - self.weights[1]) as f32
        };

        triangle[0].slerp(triangle[1], edge_weight_v0_v1).slerp(
            triangle[2].slerp(triangle[1], edge_weight_v2_v1),
            mid_weight,
        ) * self.length
    }

    /// If this coordinate is on a corner, return its index
    pub fn corner(self) -> Option<usize> {
        self.weights
            .to_array()
            .into_iter()
            .position(|w| w == self.denominator)
    }

    /// If this coordinate lies along an edge, return the indices of the two corners and the
    /// relative position along the edge.
    pub fn edge(self) -> Option<(usize, usize, Ratio<u32>)> {
        self.weights
            .to_array()
            .into_iter()
            .position(|w| w == 0)
            .map(|i| {
                let i0 = (i + 1) % 3;
                let i1 = (i + 2) % 3;
                let w = Ratio::new(self.weights[i1], self.denominator);

                (i0, i1, w)
            })
    }

    /// Linearly interpolate between the two coordinates.  
    pub fn lerp(&self, other: &Self, ratio: Ratio<u32>) -> Self {
        assert!(
            ratio.numer() <= ratio.denom(),
            "Only interpolation in 0-1 is supported, got {:?}",
            ratio
        );
        assert_eq!(self.denominator, other.denominator);

        let diff = other.weights.as_ivec3() - self.weights.as_ivec3();
        assert!(
            diff.to_array()
                .iter()
                .all(|p| p.is_multiple_of(&(*ratio.denom() as i32))),
            "Ratio not on grid"
        );

        let weights = (self.weights.as_ivec3()
            + diff * *ratio.numer() as i32 / *ratio.denom() as i32)
            .as_uvec3();
        let length = self.length.lerp(other.length, ratio.to_f32().unwrap());

        Self {
            weights,
            denominator: self.denominator,
            length,
        }
    }

    /// Convert to a mesh index for the triangular grid
    pub fn to_mesh_index(self) -> u32 {
        let a = self.denominator + 1;
        let offset = ((2 * a + 1 - self.weights.z) * self.weights.z) / 2;
        offset + self.weights.x
    }

    /// Upsample by doubling the resolution of the grid n times
    pub fn upsample(self, subdivisions: u32) -> Self {
        Self::new(self.weights.map(|d| d << subdivisions), self.length)
    }

    /// Downsample by halfing the resolution of the grid n times.
    /// Must be downsampleable without loss of precision
    pub fn downsample(self, subdivisions: u32) -> Option<Self> {
        if self
            .weights
            .to_array()
            .iter()
            .any(|d| d.trailing_zeros() < subdivisions)
        {
            return None;
        }

        let weights = self.weights.map(|d| d >> subdivisions);

        Some(Self::new(weights, self.length))
    }

    /// Move the coordinate by the given difference
    /// Diff must sum to 0
    /// If shifting would result in the coordinate being outside the triangle, None is returned.
    pub fn checked_add(self, diff: IVec3) -> Option<Self> {
        assert_eq!(diff.element_sum(), 0, "diff must sum to 0, got {:?}", diff);

        let new_weights = self.weights.as_ivec3() + diff;
        if new_weights
            .to_array()
            .iter()
            .any(|d| !(0..=self.denominator as i32).contains(d))
        {
            return None;
        }

        Some(Self::new(new_weights.as_uvec3(), self.length))
    }

    /// Convert to ratios that sum to 1
    pub fn as_ratios(&self) -> [Ratio<u32>; 3] {
        self.weights
            .to_array()
            .map(|n| Ratio::new(n, self.denominator))
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::GOLDEN_RATIO;

    use glam::Vec3A;

    use crate::coordinates::bary_iterative::BaryIterative;

    #[test]
    fn test() {
        let triangle = [
            Vec3A::new(0., 1., GOLDEN_RATIO),
            Vec3A::new(1., GOLDEN_RATIO, 0.),
            Vec3A::new(GOLDEN_RATIO, 0., 1.),
        ]
        .map(|v| v.normalize());
        println!("{:?}", triangle);

        let points_good = [
            triangle[0],                         // Corner
            triangle[0].slerp(triangle[1], 0.5), // Edge midpoint
            triangle[0].slerp(triangle[1], 0.25), // Edge 1/4
                                                 // triangle.iter().sum::<Vec3A>().normalize(), // Centre
        ];

        let points_bad = [
            -triangle[0],                                            // Corner opposite
            (triangle[0] + (triangle[0] - triangle[1])).normalize(), // Edge outside
            -triangle.iter().sum::<Vec3A>().normalize(),             // Centre opposite
        ];
        let subdivisions = 4;
        let num_sections = 1 << subdivisions;

        let barys_good = [
            // Good points
            BaryIterative {
                // Corner
                weights: [num_sections, 0, 0].into(),
                denominator: num_sections,
                length: 1.,
            },
            BaryIterative {
                // Edge midpoint
                weights: [num_sections / 2, num_sections / 2, 0].into(),
                denominator: num_sections,
                length: 1.,
            },
            BaryIterative {
                // Edge 1/4
                weights: [num_sections * 3 / 4, num_sections / 4, 0].into(),
                denominator: num_sections,
                length: 1.,
            },
            // BaryIterative {
            //     // Centre - gets rounded to nearest grid point
            //     weights: [0.3125, 0.3125, 0.375].into(),
            //     length: 1.,
            // },
        ];

        for (point, expected) in points_good.into_iter().zip(barys_good) {
            println!("\n\n");
            println!("{:?}", point);
            let bary = BaryIterative::from_cartesian(triangle, point, 4).unwrap();
            println!("{:?} -> {:?}", point, bary);

            let recon = bary.to_cartesian(triangle);
            assert!(
                point.abs_diff_eq(recon, 0.01),
                "Reconstruction failed: {:?} -> {:?}",
                point,
                recon
            );

            assert_eq!(
                bary, expected,
                "{:?} -> {:?} != {:?}",
                point, bary, expected
            );
        }

        for point in points_bad {
            let bary = BaryIterative::from_cartesian(triangle, point, 4);
            assert!(bary.is_none(), "{bary:?}");
        }
    }
}
