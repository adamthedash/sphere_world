use glam::Vec3A;

use crate::math::{EPS, almost_equal};

#[derive(Debug, Clone, Copy)]
pub struct BaryArea {
    weights: Vec3A,
    length: f32,
}

impl PartialEq for BaryArea {
    fn eq(&self, other: &Self) -> bool {
        self.weights.abs_diff_eq(other.weights, EPS) && almost_equal(self.length, other.length)
    }
}

impl BaryArea {
    pub fn from_cartesian(triangle: [Vec3A; 3], point: Vec3A) -> Self {
        for v in triangle {
            assert!(v.is_normalized(), "Traingle must be geocentric");
        }
        let [a, b, c] = triangle;

        // Normalize all points onto the unit sphere
        let (p, length) = point.normalize_and_length();

        // Compute spherical triangle area using l'Huilier's theorem.
        // Returns the spherical excess E (solid angle subtended).
        let spherical_area = |p1: Vec3A, p2: Vec3A, p3: Vec3A| -> f32 {
            // Angular side lengths (great-circle arcs)
            let a = p2.dot(p3).acos(); // side opposite p1
            let b = p1.dot(p3).acos(); // side opposite p2
            let c = p1.dot(p2).acos(); // side opposite p3
            let s = (a + b + c) / 2.0; // semi-perimeter

            // l'Huilier's theorem: tan(E/4) = sqrt(tan(s/2)*tan((s-a)/2)*tan((s-b)/2)*tan((s-c)/2))
            let t = (s / 2.0).tan()
                * ((s - a) / 2.0).tan()
                * ((s - b) / 2.0).tan()
                * ((s - c) / 2.0).tan();

            4.0 * t.max(0.0).sqrt().atan()
        };

        // Each barycentric weight = area of the sub-triangle opposite that vertex
        let w_a = spherical_area(p, b, c); // weight for vertex A
        let w_b = spherical_area(a, p, c); // weight for vertex B
        let w_c = spherical_area(a, b, p); // weight for vertex C

        // Determine sign: if P is outside the triangle, sub-areas sum > total area
        // and coordinates should be signed. Use mixed-product sign tests.
        let sign = |v1: Vec3A, v2: Vec3A| -> f32 {
            if v1.cross(v2).dot(p) >= 0.0 {
                1.0
            } else {
                -1.0
            }
        };

        let total = spherical_area(a, b, c);
        let weights = [
            sign(b, c) * w_a / total,
            sign(c, a) * w_b / total,
            sign(a, b) * w_c / total,
        ]
        .into();

        Self { weights, length }
    }

    pub fn to_cartesian(self, triangle: [Vec3A; 3]) -> Vec3A {
        unimplemented!("non-trivial")
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::GOLDEN_RATIO;

    use glam::Vec3A;

    use crate::{coordinates::bary_area::BaryArea, math::EPS};

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
            BaryArea {
                weights: todo!(),
                length: todo!(),
            },
            BaryArea {
                weights: todo!(),
                length: todo!(),
            },
        ];

        for (point, expected) in points.into_iter().zip(expecteds) {
            let bary = BaryArea::from_cartesian(triangle, point);

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
