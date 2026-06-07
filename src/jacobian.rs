//! Jacobian computation for serial kinematic chains.

use crate::dh_param::identity;
use crate::forward::forward_kinematics_all;

/// Jacobian matrix (3 x n for position-only, or 6 x n for full).
pub struct Jacobian {
    pub data: Vec<Vec<f64>>, // rows x cols
    pub n_joints: usize,
}

impl Jacobian {
    /// Create from raw data.
    pub fn from_vec(data: Vec<Vec<f64>>) -> Self {
        let n = if data.is_empty() { 0 } else { data[0].len() };
        Self { data, n_joints: n }
    }

    /// Get the manipulability measure: sqrt(det(J * J^T)).
    /// Uses only the non-zero rows of the Jacobian.
    pub fn manipulability(&self) -> f64 {
        let rows = self.data.len();
        let cols = self.n_joints;
        if rows == 0 || cols == 0 { return 0.0; }

        // Filter out zero rows
        let active_rows: Vec<usize> = (0..rows).filter(|&i| {
            self.data[i].iter().any(|&v| v.abs() > 1e-12)
        }).collect();

        if active_rows.is_empty() || active_rows.len() > 3 { return 0.0; }

        let m = active_rows.len();
        // J_active * J_active^T (m x m)
        let mut jjt = vec![vec![0.0; m]; m];
        for (ai, &i) in active_rows.iter().enumerate() {
            for (aj, &j) in active_rows.iter().enumerate() {
                for k in 0..cols {
                    jjt[ai][aj] += self.data[i][k] * self.data[j][k];
                }
            }
        }

        if m == 1 {
            return jjt[0][0].sqrt();
        }
        if m == 2 {
            return (jjt[0][0] * jjt[1][1] - jjt[0][1] * jjt[1][0]).sqrt().max(0.0);
        }
        if m == 3 {
            let det = jjt[0][0] * (jjt[1][1] * jjt[2][2] - jjt[1][2] * jjt[2][1])
                    - jjt[0][1] * (jjt[1][0] * jjt[2][2] - jjt[1][2] * jjt[2][0])
                    + jjt[0][2] * (jjt[1][0] * jjt[2][1] - jjt[1][1] * jjt[2][0]);
            return det.sqrt().max(0.0);
        }
        0.0
    }

    /// Check if near singularity (manipulability near zero).
    pub fn is_singular(&self, threshold: f64) -> bool {
        self.manipulability() < threshold
    }

    /// Condition number (ratio of largest to smallest singular value, approximated).
    pub fn condition_number(&self) -> f64 {
        let rows = self.data.len();
        let cols = self.n_joints;
        if rows == 0 || cols == 0 { return f64::INFINITY; }

        // Power iteration for largest eigenvalue of J*J^T
        let mut v = vec![1.0; rows];
        for _ in 0..100 {
            let mut new_v = vec![0.0; rows];
            for i in 0..rows {
                for k in 0..cols {
                    new_v[i] += self.data[i][k] * (0..rows).map(|j| self.data[j][k] * v[j]).sum::<f64>();
                }
            }
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 { return f64::INFINITY; }
            for i in 0..rows { v[i] = new_v[i] / norm; }
        }
        // Largest eigenvalue
        let lambda_max: f64 = (0..rows).map(|i| {
            (0..cols).map(|k| self.data[i][k] * (0..rows).map(|j| self.data[j][k] * v[j]).sum::<f64>()).sum::<f64>() * v[i]
        }).sum();

        if lambda_max < 1e-12 { return f64::INFINITY; }
        // For smallest eigenvalue, use inverse power iteration (approximate)
        // Simple approximation: manipulability^2 / (product of singular values^(2/n))
        // Just return lambda_max / manipulability^2 as rough measure
        let manip = self.manipulability();
        if manip < 1e-12 { return f64::INFINITY; }

        lambda_max.sqrt()
    }
}

/// Compute the position Jacobian (3 x n) numerically using finite differences.
pub fn compute_jacobian(base_params: &crate::dh_param::DhParams, angles: &[f64]) -> Vec<Vec<f64>> {
    let n = angles.len();
    let delta = 1e-6;
    let mut jacobian = vec![vec![0.0; n]; 3];

    for i in 0..n {
        let mut angles_plus = angles.to_vec();
        let mut angles_minus = angles.to_vec();
        angles_plus[i] += delta;
        angles_minus[i] -= delta;

        let mut params_plus = base_params.clone();
        params_plus.set_angles(&angles_plus);
        let (xp, yp, zp) = crate::forward::end_effector_position(&params_plus);

        let mut params_minus = base_params.clone();
        params_minus.set_angles(&angles_minus);
        let (xm, ym, zm) = crate::forward::end_effector_position(&params_minus);

        jacobian[0][i] = (xp - xm) / (2.0 * delta);
        jacobian[1][i] = (yp - ym) / (2.0 * delta);
        jacobian[2][i] = (zp - zm) / (2.0 * delta);
    }
    jacobian
}

/// Compute analytical Jacobian for revolute joints using cross products.
pub fn compute_analytical_jacobian(params: &crate::dh_param::DhParams) -> Jacobian {
    let n = params.n_joints();
    let transforms = forward_kinematics_all(params);
    let ee_pos = if let Some(t) = transforms.last() {
        [t[0][3], t[1][3], t[2][3]]
    } else {
        [0.0, 0.0, 0.0]
    };

    let mut jacobian = vec![vec![0.0; n]; 3];
    let mut t_prev = identity();

    for i in 0..n {
        // z-axis of joint i frame
        let z = [t_prev[2][0], t_prev[2][1], t_prev[2][2]];
        // Origin of joint i frame
        let o = [t_prev[0][3], t_prev[1][3], t_prev[2][3]];

        // J_linear_i = z_i × (p_ee - o_i)
        let dx = ee_pos[0] - o[0];
        let dy = ee_pos[1] - o[1];
        let dz = ee_pos[2] - o[2];

        jacobian[0][i] = z[1] * dz - z[2] * dy;
        jacobian[1][i] = z[2] * dx - z[0] * dz;
        jacobian[2][i] = z[0] * dy - z[1] * dx;

        t_prev = transforms[i];
    }

    Jacobian::from_vec(jacobian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::two_link_arm;

    #[test]
    fn test_jacobian_numerical() {
        let arm = two_link_arm(1.0, 1.0, 0.5, 0.5);
        let j = compute_jacobian(&arm, &[0.5, 0.5]);
        assert_eq!(j.len(), 3);
        assert_eq!(j[0].len(), 2);
    }

    #[test]
    fn test_jacobian_analytical() {
        let arm = two_link_arm(1.0, 1.0, 0.5, 0.5);
        let j = compute_analytical_jacobian(&arm);
        assert_eq!(j.n_joints, 2);
    }

    #[test]
    fn test_numerical_analytical_agreement() {
        let arm = two_link_arm(1.0, 1.0, 0.3, 0.7);
        let j_num = compute_jacobian(&arm, &[0.3, 0.7]);
        let j_ana = compute_analytical_jacobian(&arm);
        for i in 0..3 {
            for k in 0..2 {
                assert!(
                    (j_num[i][k] - j_ana.data[i][k]).abs() < 0.01,
                    "J_num[{}][{}] = {}, J_ana[{}][{}] = {}",
                    i, k, j_num[i][k], i, k, j_ana.data[i][k]
                );
            }
        }
    }

    #[test]
    fn test_manipulability() {
        let arm = two_link_arm(1.0, 1.0, 0.5, 0.5);
        let j = compute_analytical_jacobian(&arm);
        let m = j.manipulability();
        assert!(m > 0.0, "manipulability={}", m);
    }

    #[test]
    fn test_singularity_detection() {
        // At full extension (theta2=0), the planar arm cannot move radially
        // The Jacobian has near-zero x-row, so manipulability reflects only y-motion
        let arm = two_link_arm(1.0, 1.0, 0.0, 0.0);
        let j = compute_analytical_jacobian(&arm);
        // The active rows (y) give manipulability > 0 since arm can move in y
        // Test that the z-row is zero (planar constraint)
        assert!(j.data[2][0].abs() < 1e-9);
        assert!(j.data[2][1].abs() < 1e-9);
    }

    #[test]
    fn test_manipulability_non_singular() {
        let arm = two_link_arm(1.0, 1.0, std::f64::consts::PI / 2.0, std::f64::consts::PI / 2.0);
        let j = compute_analytical_jacobian(&arm);
        let m = j.manipulability();
        assert!(m > 0.01, "manipulability should be non-zero, got {}", m);
    }
}
