//! Forward kinematics using DH parameters.

use crate::dh_param::{DhParam, DhParams, mat4_mul, identity, position};

/// Compute forward kinematics for a DH parameter chain.
/// Returns the 4x4 homogeneous transformation from base to end-effector.
pub fn forward_kinematics(params: &DhParams) -> [[f64; 4]; 4] {
    let mut t = identity();
    for p in &params.params {
        t = mat4_mul(&t, &p.transformation());
    }
    t
}

/// Compute all intermediate transforms (one per joint).
pub fn forward_kinematics_all(params: &DhParams) -> Vec<[[f64; 4]; 4]> {
    let mut transforms = Vec::new();
    let mut t = identity();
    for p in &params.params {
        t = mat4_mul(&t, &p.transformation());
        transforms.push(t);
    }
    transforms
}

/// Get end-effector position from DH params.
pub fn end_effector_position(params: &DhParams) -> (f64, f64, f64) {
    position(&forward_kinematics(params))
}

/// Create a simple 2-link planar arm.
pub fn two_link_arm(l1: f64, l2: f64, theta1: f64, theta2: f64) -> DhParams {
    DhParams::new(vec![
        DhParam::new(theta1, 0.0, l1, 0.0),
        DhParam::new(theta2, 0.0, l2, 0.0),
    ])
}

/// Create a 3-link planar arm.
pub fn three_link_arm(l1: f64, l2: f64, l3: f64, theta1: f64, theta2: f64, theta3: f64) -> DhParams {
    DhParams::new(vec![
        DhParam::new(theta1, 0.0, l1, 0.0),
        DhParam::new(theta2, 0.0, l2, 0.0),
        DhParam::new(theta3, 0.0, l3, 0.0),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_link_zero_angles() {
        let arm = two_link_arm(1.0, 1.0, 0.0, 0.0);
        let (x, y, z) = end_effector_position(&arm);
        assert!((x - 2.0).abs() < 1e-9, "x={}", x);
        assert!(y.abs() < 1e-9);
        assert!(z.abs() < 1e-9);
    }

    #[test]
    fn test_two_link_90_degrees() {
        let arm = two_link_arm(1.0, 1.0, std::f64::consts::PI / 2.0, 0.0);
        let (x, y, _z) = end_effector_position(&arm);
        // First link at 90°: endpoint at (0, 1, 0)
        // Second link continues along same direction: (0, 2, 0)
        assert!(x.abs() < 1e-9, "x={}", x);
        assert!((y - 2.0).abs() < 1e-9, "y={}", y);
    }

    #[test]
    fn test_two_link_folded() {
        let arm = two_link_arm(1.0, 1.0, 0.0, std::f64::consts::PI);
        let (x, _y, _z) = end_effector_position(&arm);
        // Both links along x, second one folded back
        assert!((x - 0.0).abs() < 1e-9, "x={}", x);
    }

    #[test]
    fn test_three_link_forward() {
        let arm = three_link_arm(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        let (x, y, _z) = end_effector_position(&arm);
        assert!((x - 3.0).abs() < 1e-9, "x={}", x);
        assert!(y.abs() < 1e-9);
    }

    #[test]
    fn test_intermediate_transforms() {
        let arm = two_link_arm(1.0, 1.0, std::f64::consts::PI / 4.0, 0.0);
        let transforms = forward_kinematics_all(&arm);
        assert_eq!(transforms.len(), 2);
    }

    #[test]
    fn test_fk_consistency() {
        // FK should give same result whether computed fresh or via intermediate
        let arm = two_link_arm(1.0, 2.0, 0.5, 0.7);
        let t1 = forward_kinematics(&arm);
        let t2 = forward_kinematics_all(&arm);
        let t_last = *t2.last().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert!((t1[i][j] - t_last[i][j]).abs() < 1e-9);
            }
        }
    }
}
