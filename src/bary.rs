use glam::{Mat3A, UVec3, Vec2, Vec3A};
use itertools::Itertools;
use num::{Integer, ToPrimitive, rational::Ratio};

use crate::math::almost_equal;

#[derive(Debug, Clone, Copy)]
pub struct Barycentric {
    /// Sum normalised to 1
    /// Distances from edge opposite vertices [2, 0, 1]
    pub distances: [f32; 3],
    /// Signed vector length through origin
    pub length: f32,
}

impl PartialEq for Barycentric {
    fn eq(&self, other: &Self) -> bool {
        self.distances
            .iter()
            .zip(&other.distances)
            .all(|(a, b)| almost_equal(*a, *b))
            && almost_equal(self.length, other.length)
    }
}

impl Barycentric {
    /// Snap to the vectex grid of the given size.
    /// Out of bounds returns None
    pub fn snap_even(self, num_subdivisions: u32) -> Option<BarycentricSnapped> {
        if self.distances.iter().any(|d| d.is_nan() || d.is_infinite()) {
            return None;
        }

        let denom = 2_u32.pow(num_subdivisions);
        let numerator = self.distances.map(|d| d * denom as f32);

        // Custom rounding - Fractional weight is assigned to closest axes
        let mut ints = numerator.map(|n| n.trunc() as i32);
        let mut fracts = numerator.map(|n| n.fract());
        let total_fracts = fracts.iter().sum::<f32>().round() as u32;
        for _ in 0..total_fracts {
            // Assign the full weight to closest point
            let i = fracts
                .iter()
                .position_max_by(|a, b| a.total_cmp(b))
                .unwrap();
            ints[i] += 1;
            fracts[i] -= 1.;
        }

        let numerator = ints;

        if !numerator.iter().all(|n| (0..=denom as i32).contains(n)) {
            return None;
        }

        let distances = UVec3::from_array(numerator.map(|n| n as u32));

        assert_eq!(
            distances.element_sum(),
            denom,
            "Bary distances must sum to 1, converted {:?} -> {:?} nsubs: {:?}",
            self,
            (distances, denom),
            num_subdivisions
        );
        Some(BarycentricSnapped::new(distances, self.length))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarycentricSnapped {
    /// Sum normalised to 1
    /// Distances from edge opposite vertices [2, 0, 1]
    pub distances: UVec3,

    pub denominator: u32,
    /// Signed vector length through origin
    pub length: f32,
}

impl BarycentricSnapped {
    pub fn new(distances: UVec3, length: f32) -> Self {
        let denominator = distances.element_sum();
        Self {
            distances,
            denominator,
            length,
        }
    }

    pub fn as_ratios(&self) -> [Ratio<u32>; 3] {
        self.distances
            .to_array()
            .map(|n| Ratio::new(n, self.denominator))
    }

    /// Linearly interpolate between the two coordinates.  
    pub fn interp(&self, other: &Self, ratio: Ratio<u32>) -> Self {
        assert!(
            ratio.numer() <= ratio.denom(),
            "Only interpolation in 0-1 is supported"
        );
        assert_eq!(self.denominator, other.denominator);

        let diff = other.distances.as_ivec3() - self.distances.as_ivec3();
        assert!(
            diff.to_array()
                .iter()
                .all(|p| p.is_multiple_of(&(*ratio.denom() as i32))),
            "Ratio not on grid"
        );

        let distances = (self.distances.as_ivec3()
            + diff * *ratio.numer() as i32 / *ratio.denom() as i32)
            .as_uvec3();
        let length = self.length * ratio.to_f32().unwrap();

        Self {
            distances,
            denominator: self.denominator,
            length,
        }
    }
}

/// A 1d point along the edge of one pentagon
/// Between 0 and 1
#[derive(Debug)]
struct EdgeLength {
    /// The number of sections along
    /// x <= l
    x: u32,
    /// The number of sections this edge is divided into
    /// l >= 1
    /// l is a power of 2
    l: u32,
}

#[derive(Debug)]
struct TriangularCoord {
    /// Basis vectors formed by taking two edges along the face of a triangle
    basis: Mat3A,
    /// Lengths along the basis vectors
    uv: [EdgeLength; 2],
}

fn cartesian_to_triangular(
    triangle: [Vec3A; 3],
    point: Vec3A,
    num_segments: u32,
) -> TriangularCoord {
    let basis = Mat3A::from_cols(
        triangle[1] - triangle[0],
        triangle[2] - triangle[0],
        triangle[0],
    );
    let projected = (basis.inverse() * point).truncate();

    let Vec2 {
        x: u_length,
        y: v_length,
    } = projected;

    let u_length = EdgeLength {
        x: (u_length * num_segments as f32).round() as u32,
        l: num_segments,
    };
    let v_length = EdgeLength {
        x: (v_length * num_segments as f32).round() as u32,
        l: num_segments,
    };

    TriangularCoord {
        basis,
        uv: [u_length, v_length],
    }
}

pub fn cartesian_to_barycentric(triangle: [Vec3A; 3], point: Vec3A) -> Barycentric {
    let distances = triangle
        .iter()
        .circular_tuple_windows()
        .map(|(v0, v1, v2)| {
            let midpoint = v0.midpoint(*v1);
            let height = v2 - midpoint;
            let mid_p = point - midpoint;

            // Project through point onto triangle face
            let basis = Mat3A::from_cols(point.normalize(), height, (v1 - v0).normalize());
            let projected = basis.inverse() * mid_p;

            projected.y
        })
        .collect_array::<3>()
        .unwrap();

    let midpoint = triangle.iter().sum::<Vec3A>() / 3.;
    let length = midpoint.normalize().dot(point) / midpoint.length();

    Barycentric { distances, length }
}

#[cfg(test)]
mod tests {
    use std::{assert_matches, f32::consts::GOLDEN_RATIO};

    use glam::{Mat3A, UVec3, Vec3A};
    use num::rational::Ratio;

    use crate::bary::{
        Barycentric, BarycentricSnapped, EdgeLength, TriangularCoord, cartesian_to_barycentric,
        cartesian_to_triangular,
    };

    #[test]
    fn test_triangular() {
        let triangle = [
            Vec3A::new(0., 1., GOLDEN_RATIO),
            Vec3A::new(1., GOLDEN_RATIO, 0.),
            Vec3A::new(GOLDEN_RATIO, 0., 1.),
        ];

        let num_segments = 2;

        let point = triangle[0];
        let triangular = cartesian_to_triangular(triangle, point, num_segments);
        assert_matches!(
            triangular,
            TriangularCoord {
                uv: [EdgeLength { x: 0, l: 2 }, EdgeLength { x: 0, l: 2 }],
                ..
            }
        );

        let point = triangle[1];
        let triangular = cartesian_to_triangular(triangle, point, num_segments);
        assert_matches!(
            triangular,
            TriangularCoord {
                uv: [EdgeLength { x: 2, l: 2 }, EdgeLength { x: 0, l: 2 }],
                ..
            }
        );

        let point = triangle[2];
        let triangular = cartesian_to_triangular(triangle, point, num_segments);
        assert_matches!(
            triangular,
            TriangularCoord {
                uv: [EdgeLength { x: 0, l: 2 }, EdgeLength { x: 2, l: 2 }],
                ..
            }
        );

        let point = Vec3A::X + Vec3A::Y + Vec3A::Z;
        let triangular = cartesian_to_triangular(triangle, point, num_segments);
        assert_matches!(
            triangular,
            TriangularCoord {
                uv: [EdgeLength { x: 1, l: 2 }, EdgeLength { x: 1, l: 2 }],
                ..
            }
        );
    }

    #[test]
    fn test_barycentric() {
        let triangle = [
            Vec3A::new(0., 1., GOLDEN_RATIO),
            Vec3A::new(1., GOLDEN_RATIO, 0.),
            Vec3A::new(GOLDEN_RATIO, 0., 1.),
        ];

        let point = triangle.iter().sum();
        let bary = cartesian_to_barycentric(triangle, point);
        assert_eq!(
            bary,
            Barycentric {
                distances: [1. / 3., 1. / 3., 1. / 3.],
                length: 3.
            }
        );

        let point = (triangle[0] + triangle[1]) * 1.;
        let bary = cartesian_to_barycentric(triangle, point);
        assert_eq!(
            bary,
            Barycentric {
                distances: [0., 1. / 2., 1. / 2.],
                length: 2.
            }
        );
    }

    #[test]
    fn test_snap() {
        // Case was failing due to snapped coords not summing to 1
        let bary = Barycentric {
            distances: [0.033546187, 0.6403883, 0.32606557],
            length: 1.1677722,
        };
        let snapped = bary.snap_even(5).unwrap();
        assert_eq!(
            snapped,
            BarycentricSnapped::new(UVec3::new(1, 21, 10), bary.length)
        );

        // Case was failing due to snapped coords not summing to 1
        let bary = Barycentric {
            distances: [5.5879354e-9, 0.49999994, 0.4999999],
            length: 1.0422658,
        };
        let snapped = bary.snap_even(2).unwrap();
        assert_eq!(
            snapped,
            BarycentricSnapped::new(UVec3::new(0, 2, 2), bary.length)
        );
    }

    #[test]
    fn test_interp() {
        let b0 = BarycentricSnapped::new([0, 0, 16].into(), 1.);
        let b1 = BarycentricSnapped::new([16, 0, 0].into(), 1.);
        let b = b0.interp(&b1, Ratio::ZERO);
        assert_eq!(b.distances.to_array(), [0, 0, 16]);
        let b = b0.interp(&b1, Ratio::ONE);
        assert_eq!(b.distances.to_array(), [16, 0, 0]);
        let b = b0.interp(&b1, Ratio::new(1, 2));
        assert_eq!(b.distances.to_array(), [8, 0, 8]);
    }
}
