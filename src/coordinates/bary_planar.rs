use glam::{Mat3A, Vec3A, Vec3Swizzles};

use crate::math::{EPS, almost_equal};

/// Defined by the 3 great circle planes passing through each triangle edge.
/// Weights are the (normalised) angles that the planes are rotated towards the opposite corner
/// Reconstruction is done by rotating the planes and finding their intersecting vector.
/// A combination of any 2 are enough to reconstruct.
/// (need to verify) As angles are used directly as weights, areas of subdivsions should be
/// maintained.
#[derive(Debug, Clone, Copy)]
pub struct BaryPlanar {
    /// Normalised (0-1) rotation angle of great plane towards this corner
    /// [0] -> plane of v1-v2 rotated towards v0
    angles: Vec3A,
    length: f32,
}

impl PartialEq for BaryPlanar {
    fn eq(&self, other: &Self) -> bool {
        self.angles.abs_diff_eq(other.angles, EPS) && almost_equal(self.length, other.length)
    }
}

impl BaryPlanar {
    pub fn from_cartesian(triangle: [Vec3A; 3], point: Vec3A) -> Self {
        for v in triangle {
            assert!(v.is_normalized(), "Traingle must be geocentric");
        }

        let (point, length) = point.normalize_and_length();

        let is_ccw = triangle[0].cross(triangle[1]).dot(triangle[2]) > 0.;
        assert!(!is_ccw, "assuming CW for now");

        let mut angles = [0.; 3];
        for i in 0..3 {
            let mut t = triangle;
            t.rotate_left(i);
            let [v0, v1, v2] = t;

            // Basis vectors of plane
            let normal = v2.cross(v1).normalize();
            let forward = (v1 - v2).normalize();
            let up = forward.cross(normal).normalize();
            let basis = Mat3A::from_cols(forward, normal, up);

            // Project the point onto the plane
            let projected = point - point.dot(normal) * normal;
            // NOTE: this won't work if point is > 90 deg from plane
            let angle = normal.dot(point).asin();
            let total_angle = normal.dot(v0).asin();

            // // Find angle between plane and opposing vertex
            // let angle = (basis.inverse() * point).zy().to_angle();
            //
            // // Normalise to 0-1
            // let total_angle = (basis.inverse() * v0).zy().to_angle();

            angles[i] = angle / total_angle;
        }

        // Normalise to sum to 1 - needed?
        let sum = angles.iter().sum::<f32>();
        angles.iter_mut().for_each(|a| *a /= sum);

        Self {
            angles: angles.into(),
            length,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::GOLDEN_RATIO;

    use glam::Vec3A;

    use super::BaryPlanar;
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
            triangle[0].slerp(triangle[1], 0.75),       // Edge 3/4
            triangle.iter().sum::<Vec3A>().normalize(), // Centre
            // Bad points
            -triangle[0],                                            // Corner opposite
            (triangle[0] + (triangle[0] - triangle[1])).normalize(), // Edge outside
            -triangle.iter().sum::<Vec3A>().normalize(),             // Centre opposite
        ];

        let expecteds = [
            // Good points
            BaryPlanar {
                // Corner
                angles: [1., 0., 0.].into(),
                length: 1.,
            },
            BaryPlanar {
                // Edge midpoint
                angles: [0.5, 0.5, 0.].into(),
                length: 1.,
            },
            BaryPlanar {
                // Edge 1/4
                angles: [0.75, 0.25, 0.].into(),
                length: 1.,
            },
            BaryPlanar {
                // Edge 1/4
                angles: [0.25, 0.75, 0.].into(),
                length: 1.,
            },
            BaryPlanar {
                // Centre
                angles: [1. / 3.; 3].into(),
                length: 1.,
            },
            // Bad points
            BaryPlanar {
                // Corner opposite
                angles: [-1., 0., 0.].into(),
                length: 1.,
            },
            BaryPlanar {
                // Edge outside
                angles: [-1., 0., 0.].into(),
                length: 1.,
            },
            BaryPlanar {
                // Centre opposite
                angles: [-1., 0., 0.].into(),
                length: 1.,
            },
        ];

        for (point, expected) in points.into_iter().zip(expecteds) {
            let bary = BaryPlanar::from_cartesian(triangle, point);

            // let recon = bary.to_cartesian(triangle);
            // assert!(
            //     point.abs_diff_eq(recon, 0.01),
            //     "Reconstruction failed: {:?} -> {:?}",
            //     point,
            //     recon
            // );

            assert_eq!(
                bary, expected,
                "{:?} -> {:?} != {:?}",
                point, bary, expected
            );
        }
    }
}
