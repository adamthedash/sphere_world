use glam::Vec3A;

use crate::math::{EPS, almost_equal};

/// P = v0.slerp(v1, s/(1-t)).slerp(v2, t) * length
#[derive(Debug, Clone, Copy)]
pub struct BarySlerp {
    // Sum to 1
    weights: Vec3A,
    // Vector length
    length: f32,
}

impl PartialEq for BarySlerp {
    fn eq(&self, other: &Self) -> bool {
        // Since cartesian -> bary conversion is iterative, accept a higher tolerance as it's more
        // than just floating point error
        self.weights.abs_diff_eq(other.weights, 0.01) && almost_equal(self.length, other.length)
    }
}

impl BarySlerp {
    pub fn from_cartesian(triangle: [Vec3A; 3], point: Vec3A) -> Self {
        for v in triangle {
            assert!(v.is_normalized(), "Traingle must be geocentric");
        }

        let [a, b, c] = triangle;
        let (p, length) = point.normalize_and_length();

        // Vertex degeneracy: if P coincides with a vertex, return exact corner weights
        if let Some(i) = triangle.iter().position(|v| v.abs_diff_eq(p, EPS)) {
            let mut weights = [0.; 3];
            weights[i] = 1.;
            return Self {
                weights: weights.into(),
                length,
            };
        }

        // Normal of the great circle through C and P
        let n = c.cross(p).normalize();

        // Find s: root of f(s) = n . slerp(A, B, s) = 0  via bisection
        let f = |s: f32| -> f32 { n.dot(a.slerp(b, s)) };

        let (f0, f1) = (f(0.0), f(1.0));

        let s = if almost_equal(f0, 0.) {
            0.0
        } else if almost_equal(f1, 0.) {
            1.0
        } else if f0 * f1 > 0.0 {
            // Foot is outside [0,1] — clamp to nearest endpoint
            if f0.abs() < f1.abs() { 0.0 } else { 1.0 }
        } else {
            let (mut lo, mut hi) = (0.0f32, 1.0f32);
            for _ in 0..64 {
                let mid = (lo + hi) * 0.5;
                if f(mid) * f0 < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
                if hi - lo < EPS {
                    break;
                }
            }
            (lo + hi) * 0.5
        };

        // Q is the foot-point on arc AB
        let q = a.slerp(b, s);

        // Find t = d(Q, P) / d(Q, C)
        let d_qp = q.dot(p).clamp(-1.0, 1.0).acos();
        let d_qc = q.dot(c).clamp(-1.0, 1.0).acos();
        let t = if d_qc > EPS {
            (d_qp / d_qc).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let wc = t;
        let wa = (1.0 - t) * (1.0 - s);
        let wb = (1.0 - t) * s;

        Self {
            weights: Vec3A::new(wa, wb, wc),
            length,
        }
    }

    pub fn to_cartesian(self, triangle: [Vec3A; 3]) -> Vec3A {
        let t = self.weights[2];
        let s = self.weights[1] / (self.weights[0] + self.weights[1]);

        triangle[0].slerp(triangle[1], s).slerp(triangle[2], t) * self.length
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::GOLDEN_RATIO;

    use glam::Vec3A;

    use super::BarySlerp;
    use crate::math::EPS;

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
            BarySlerp {
                // Corner
                weights: [1., 0., 0.].into(),
                length: 1.,
            },
            BarySlerp {
                // Edge midpoint
                weights: [0.5, 0.5, 0.].into(),
                length: 1.,
            },
            BarySlerp {
                // Edge 1/4
                weights: [0.75, 0.25, 0.].into(),
                length: 1.,
            },
            BarySlerp {
                // Centre
                weights: [1. / 3.; 3].into(),
                length: 1.,
            },
            // Bad points
            BarySlerp {
                // Corner opposite
                weights: [-1., 0., 0.].into(),
                length: 1.,
            },
            BarySlerp {
                // Edge outside
                weights: [-1., 0., 0.].into(),
                length: 1.,
            },
            BarySlerp {
                // Centre opposite
                weights: [-1., 0., 0.].into(),
                length: 1.,
            },
        ];

        for (point, expected) in points.into_iter().zip(expecteds) {
            let bary = BarySlerp::from_cartesian(triangle, point);

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
    }
}
