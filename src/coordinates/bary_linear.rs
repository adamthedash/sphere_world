use glam::Vec3A;

use crate::math::{EPS, almost_equal};

/// Represents a point relative to a geodesic triangle.
/// P = w^T * t * c * l
#[derive(Debug, Clone, Copy)]
pub struct BaryLinear {
    /// Relative weights of triangle corners.
    /// Sums to 1
    /// w^t * t => point on triangle plane
    weights: Vec3A,
    /// Scaling factor between plane and sphere's surface
    /// |w^T * t * c| == 1
    c: f32,
    /// Vector length
    length: f32,
}

impl PartialEq for BaryLinear {
    fn eq(&self, other: &Self) -> bool {
        self.weights.abs_diff_eq(other.weights, EPS)
            && almost_equal(self.c, other.c)
            && almost_equal(self.length, other.length)
    }
}

impl BaryLinear {
    /// Convert to bary coordinates on the provided geodesic triangle
    pub fn from_cartesian(triangle: [Vec3A; 3], point: Vec3A) -> Self {
        for v in triangle {
            assert!(v.is_normalized(), "Traingle must be geocentric");
        }

        let [a, b, c] = triangle;

        // Normalize all points onto the unit sphere
        let (p, length) = point.normalize_and_length();

        // Build matrix M = [A | B | C] as column vectors, then solve M * [u,v,w]^T = P.
        // Cramer's rule: u = det([P,B,C])/det([A,B,C]), etc.
        let det = |c0: Vec3A, c1: Vec3A, c2: Vec3A| -> f32 { c0.dot(c1.cross(c2)) };

        let weights = Vec3A::new(
            det(p, b, c), //
            det(a, p, c),
            det(a, b, p),
        ) / det(a, b, c);

        // Normalise to 1
        let c = weights.element_sum();

        Self {
            weights: weights / c,
            c,
            length,
        }
    }

    pub fn to_cartesian(self, triangle: [Vec3A; 3]) -> Vec3A {
        for v in triangle {
            assert!(v.is_normalized(), "Traingle must be geocentric");
        }

        self.weights
            .to_array()
            .iter()
            .zip(&triangle)
            .map(|(d, v)| *v * *d)
            .sum::<Vec3A>()
            * self.c
            * self.length
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::GOLDEN_RATIO;

    use glam::Vec3A;

    use crate::{coordinates::bary_linear::BaryLinear, math::EPS};

    #[test]
    fn test() {
        let triangle = [
            Vec3A::new(0., 1., GOLDEN_RATIO),
            Vec3A::new(1., GOLDEN_RATIO, 0.),
            Vec3A::new(GOLDEN_RATIO, 0., 1.),
        ]
        .map(|v| v.normalize());
        println!("{:?}", triangle);

        let points = [
            // Good points
            triangle[0],                                // Corner
            triangle[0].slerp(triangle[1], 0.5),        // Edge midpoint
            triangle[0].slerp(triangle[1], 0.25),       // Edge 1/4
            triangle.iter().sum::<Vec3A>().normalize(), // Centre
            // Bad points
            -triangle[0],                                            // Corner opposite
            (triangle[0] + (triangle[0] - triangle[1])).normalize(), // Edge outside
            -triangle.iter().sum::<Vec3A>().normalize(),             // Centre opposite
        ];

        let expecteds = [
            // Good points
            BaryLinear {
                // Corner
                weights: [1., 0., 0.].into(),
                c: 1.,
                length: 1.,
            },
            BaryLinear {
                // Edge midpoint
                weights: [0.5, 0.5, 0.].into(),
                c: 1. / 0.85065,
                length: 1.,
            },
            BaryLinear {
                // Edge 1/4
                weights: [0.25, 0.75, 0.].into(),
                c: 1.,
                length: 1.,
            },
            BaryLinear {
                // Centre
                weights: [1. / 3.; 3].into(),
                c: 1.,
                length: 1.,
            },
            // Bad points
            BaryLinear {
                // Corner opposite
                weights: [-1., 0., 0.].into(),
                c: 1.,
                length: 1.,
            },
            BaryLinear {
                // Edge outside
                weights: [-1., 0., 0.].into(),
                c: 1.,
                length: 1.,
            },
            BaryLinear {
                // Centre opposite
                weights: [-1., 0., 0.].into(),
                c: 1.,
                length: 1.,
            },
        ];

        for (point, expected) in points.into_iter().zip(expecteds) {
            let bary = BaryLinear::from_cartesian(triangle, point);

            let recon = bary.to_cartesian(triangle);
            assert!(
                point.abs_diff_eq(recon, EPS),
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
    }
}
