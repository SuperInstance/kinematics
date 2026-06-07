//! Denavit-Hartenberg parameter representation.

/// A single DH parameter set (standard convention).
/// Transform: Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
#[derive(Debug, Clone, Copy)]
pub struct DhParam {
    /// Joint angle / variable (theta for revolute)
    pub theta: f64,
    /// Link offset along z
    pub d: f64,
    /// Link length along x
    pub a: f64,
    /// Link twist about x
    pub alpha: f64,
}

impl DhParam {
    pub fn new(theta: f64, d: f64, a: f64, alpha: f64) -> Self {
        Self { theta, d, a, alpha }
    }

    /// Compute the 4x4 transformation matrix for this DH link.
    pub fn transformation(&self) -> [[f64; 4]; 4] {
        let ct = self.theta.cos();
        let st = self.theta.sin();
        let ca = self.alpha.cos();
        let sa = self.alpha.sin();

        [
            [ct, -st * ca,  st * sa, self.a * ct],
            [st,  ct * ca, -ct * sa, self.a * st],
            [0.0,      sa,       ca,        self.d],
            [0.0,     0.0,      0.0,        1.0],
        ]
    }

    /// Create a revolute joint DH param (theta varies).
    pub fn revolute(d: f64, a: f64, alpha: f64, theta: f64) -> Self {
        Self { theta, d, a, alpha }
    }
}

/// A set of DH parameters for a serial kinematic chain.
#[derive(Debug, Clone)]
pub struct DhParams {
    pub params: Vec<DhParam>,
}

impl DhParams {
    pub fn new(params: Vec<DhParam>) -> Self {
        Self { params }
    }

    /// Number of joints.
    pub fn n_joints(&self) -> usize {
        self.params.len()
    }

    /// Set joint angles (modifies theta for revolute joints).
    pub fn set_angles(&mut self, angles: &[f64]) {
        for (i, &angle) in angles.iter().enumerate() {
            if i < self.params.len() {
                self.params[i].theta = angle;
            }
        }
    }

    /// Get current joint angles.
    pub fn get_angles(&self) -> Vec<f64> {
        self.params.iter().map(|p| p.theta).collect()
    }
}

/// Multiply two 4x4 transformation matrices.
pub fn mat4_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut r = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

/// Extract position from a 4x4 transform.
pub fn position(t: &[[f64; 4]; 4]) -> (f64, f64, f64) {
    (t[0][3], t[1][3], t[2][3])
}

/// Extract rotation matrix from a 4x4 transform.
pub fn rotation_matrix(t: &[[f64; 4]; 4]) -> [[f64; 3]; 3] {
    [
        [t[0][0], t[0][1], t[0][2]],
        [t[1][0], t[1][1], t[1][2]],
        [t[2][0], t[2][1], t[2][2]],
    ]
}

/// Identity 4x4 transform.
pub fn identity() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let id = identity();
        assert!((id[0][0] - 1.0).abs() < 1e-9);
        assert!(id[0][3].abs() < 1e-9);
    }

    #[test]
    fn test_dh_zero_transform() {
        let dh = DhParam::new(0.0, 0.0, 0.0, 0.0);
        let t = dh.transformation();
        // All zeros → identity transform
        assert!((t[0][0] - 1.0).abs() < 1e-9);
        assert!((t[1][1] - 1.0).abs() < 1e-9);
        assert!((t[2][2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dh_translation_along_x() {
        let dh = DhParam::new(0.0, 0.0, 5.0, 0.0);
        let t = dh.transformation();
        assert!((t[0][3] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_dh_rotation_about_z() {
        let dh = DhParam::new(std::f64::consts::PI / 2.0, 0.0, 0.0, 0.0);
        let t = dh.transformation();
        // cos(pi/2) ≈ 0
        assert!(t[0][0].abs() < 1e-9);
        // sin(pi/2) ≈ 1
        assert!((t[1][0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mat4_multiply_identity() {
        let id = identity();
        let result = mat4_mul(&id, &id);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result[i][j] - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_params_set_get_angles() {
        let mut params = DhParams::new(vec![
            DhParam::revolute(0.0, 1.0, 0.0, 0.0),
            DhParam::revolute(0.0, 1.0, 0.0, 0.0),
        ]);
        params.set_angles(&[0.5, 1.0]);
        let angles = params.get_angles();
        assert!((angles[0] - 0.5).abs() < 1e-9);
        assert!((angles[1] - 1.0).abs() < 1e-9);
    }
}
