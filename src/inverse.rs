//! Inverse kinematics using Jacobian-based methods.

use crate::dh_param::DhParams;
use crate::forward::end_effector_position;
use crate::jacobian::compute_jacobian;

/// Solve inverse kinematics using the damped least-squares (Levenberg-Marquardt) Jacobian method.
/// Returns joint angles that place the end-effector at the target (x, y, z).
pub fn inverse_kinematics(
    base_params: &DhParams,
    target: (f64, f64, f64),
    initial_angles: &[f64],
    max_iterations: usize,
    tolerance: f64,
    damping: f64,
) -> Result<Vec<f64>, String> {
    let n = base_params.n_joints();
    if initial_angles.len() != n {
        return Err("Initial angles must match number of joints".to_string());
    }

    let mut angles = initial_angles.to_vec();
    let lambda_sq = damping * damping;

    for _ in 0..max_iterations {
        let mut params = base_params.clone();
        params.set_angles(&angles);

        let (x, y, z) = end_effector_position(&params);
        let error = vec![target.0 - x, target.1 - y, target.2 - z];

        let error_norm: f64 = error.iter().map(|e| e * e).sum::<f64>().sqrt();
        if error_norm < tolerance {
            return Ok(angles);
        }

        let j = compute_jacobian(base_params, &angles);

        // Damped least squares: dθ = J^T (J J^T + λ²I)^{-1} e
        // For small n, solve (J J^T + λ²I) * x = e, then dθ = J^T * x
        let jjt = mat_mul_3n_nt(&j, n);
        let mut a = jjt;
        for i in 0..3 {
            a[i][i] += lambda_sq;
        }
        let a_inv = invert_3x3(&a);
        let tmp = mat_vec_mul_3(&a_inv, &error);
        let delta = jt_vec_mul(&j, &tmp, n);

        for i in 0..n {
            angles[i] += delta[i];
        }
    }

    // Check final error
    let mut params = base_params.clone();
    params.set_angles(&angles);
    let (x, y, z) = end_effector_position(&params);
    let final_error = ((target.0 - x).powi(2) + (target.1 - y).powi(2) + (target.2 - z).powi(2)).sqrt();
    if final_error < tolerance * 10.0 {
        Ok(angles)
    } else {
        Err(format!("IK did not converge, error={}", final_error))
    }
}

/// Analytical IK for 2-link planar arm.
pub fn ik_2link(l1: f64, l2: f64, x: f64, y: f64) -> Result<(f64, f64), String> {
    let d_sq = x * x + y * y;
    let d = d_sq.sqrt();

    if d > l1 + l2 {
        return Err("Target out of reach".to_string());
    }
    if d < (l1 - l2).abs() {
        return Err("Target too close (singular)".to_string());
    }

    let cos_theta2 = (d_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2);
    let cos_theta2 = cos_theta2.clamp(-1.0, 1.0);
    let theta2 = cos_theta2.acos();

    let k1 = l1 + l2 * cos_theta2;
    let k2 = l2 * theta2.sin();
    let theta1 = y.atan2(x) - k2.atan2(k1);

    Ok((theta1, theta2))
}

fn mat_mul_3n_nt(j: &Vec<Vec<f64>>, n: usize) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for k in 0..3 {
            for l in 0..n {
                r[i][k] += j[i][l] * j[k][l];
            }
        }
    }
    r
}

fn invert_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let inv_det = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

fn mat_vec_mul_3(m: &[[f64; 3]; 3], v: &[f64]) -> Vec<f64> {
    vec![
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn jt_vec_mul(j: &Vec<Vec<f64>>, v: &Vec<f64>, n: usize) -> Vec<f64> {
    let mut r = vec![0.0; n];
    for i in 0..n {
        for k in 0..3 {
            r[i] += j[k][i] * v[k];
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::two_link_arm;

    #[test]
    fn test_analytical_ik_2link_reachable() {
        let (t1, t2) = ik_2link(1.0, 1.0, 1.0, 1.0).unwrap();
        // Verify via FK
        let arm = two_link_arm(1.0, 1.0, t1, t2);
        let (x, y, _) = crate::forward::end_effector_position(&arm);
        assert!((x - 1.0).abs() < 0.01, "x={}", x);
        assert!((y - 1.0).abs() < 0.01, "y={}", y);
    }

    #[test]
    fn test_analytical_ik_2link_out_of_reach() {
        let result = ik_2link(1.0, 1.0, 5.0, 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_jacobian_ik_2link() {
        let base = two_link_arm(1.0, 1.0, 0.0, 0.0);
        let result = inverse_kinematics(
            &base,
            (1.5, 0.5, 0.0),
            &[0.5, 0.5],
            100,
            0.01,
            0.1,
        );
        assert!(result.is_ok());
        let angles = result.unwrap();
        let mut arm = base.clone();
        arm.set_angles(&angles);
        let (x, y, _) = crate::forward::end_effector_position(&arm);
        assert!((x - 1.5).abs() < 0.1, "x={}", x);
        assert!((y - 0.5).abs() < 0.1, "y={}", y);
    }

    #[test]
    fn test_analytical_ik_roundtrip() {
        // Set random angles, compute FK, then solve IK
        let arm = two_link_arm(1.0, 1.0, 0.8, 0.6);
        let (x, y, _) = crate::forward::end_effector_position(&arm);
        let (t1, t2) = ik_2link(1.0, 1.0, x, y).unwrap();
        let arm2 = two_link_arm(1.0, 1.0, t1, t2);
        let (x2, y2, _) = crate::forward::end_effector_position(&arm2);
        assert!((x - x2).abs() < 0.01);
        assert!((y - y2).abs() < 0.01);
    }

    #[test]
    fn test_analytical_ik_full_extension() {
        let (t1, _t2) = ik_2link(1.0, 1.0, 2.0, 0.0).unwrap();
        assert!(t1.abs() < 0.01);
    }
}
