use glam::Vec3A;

#[derive(Debug, Clone, Copy)]
pub struct PolarCoord {
    /// Radians from equator
    pub lat: f32,
    /// Radians east of Grenwich meridian
    pub lon: f32,
    /// Unit sphere
    pub alt: f32,
}

impl PolarCoord {
    pub fn to_xyz(self) -> Vec3A {
        let y = self.alt * self.lat.sin();
        let eq = self.alt * self.lat.cos();

        let x = eq * self.lon.sin();
        let z = eq * self.lon.cos();

        Vec3A::new(x, y, z)
    }

    /// -Z : GMT
    /// +Y : North pole
    /// +X : UTC +6
    pub fn from_xyz(point: Vec3A) -> Self {
        let [x, y, z] = point.to_array();
        let equator_line = ((x * x) + (z * z)).sqrt();

        let lat = y.atan2(equator_line);
        let lon = x.atan2(z);
        let alt = point.length();

        Self { lat, lon, alt }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3A;

    use super::PolarCoord;

    #[test]
    fn test_polar() {
        let points = [-1., 0., 1.];
        for x in points {
            for y in points {
                for z in points {
                    let p = Vec3A::from_array([x, y, z]);
                    let transformed = PolarCoord::from_xyz(p).to_xyz();
                    assert!(
                        p.abs_diff_eq(transformed, 1e-6),
                        "{:?}, {:?}",
                        p,
                        transformed
                    );
                }
            }
        }
    }
}
