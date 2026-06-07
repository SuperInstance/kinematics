//! Workspace analysis for robotic manipulators.

use crate::dh_param::DhParams;
use crate::forward::end_effector_position;

/// Workspace analysis results.
#[derive(Debug, Clone)]
pub struct WorkspaceAnalysis {
    /// Minimum reach
    pub min_reach: f64,
    /// Maximum reach
    pub max_reach: f64,
    /// Samples used
    pub samples: usize,
    /// All sampled points (x, y, z)
    pub points: Vec<(f64, f64, f64)>,
}

/// Analyze the workspace of a kinematic chain by sampling joint angles.
pub fn analyze_workspace(
    base_params: &DhParams,
    angle_limits: &[(f64, f64)],
    samples_per_joint: usize,
) -> WorkspaceAnalysis {
    let _n = base_params.n_joints();
    let mut points = Vec::new();
    let mut min_reach = f64::INFINITY;
    let mut max_reach = f64::NEG_INFINITY;

    // Generate all angle combinations recursively
    let mut angle_sets = Vec::new();
    generate_angle_combinations(angle_limits, samples_per_joint, &mut angle_sets, &mut Vec::new());

    for angles in &angle_sets {
        let mut params = base_params.clone();
        params.set_angles(angles);
        let (x, y, z) = end_effector_position(&params);
        let reach = (x * x + y * y + z * z).sqrt();
        min_reach = min_reach.min(reach);
        max_reach = max_reach.max(reach);
        points.push((x, y, z));
    }

    WorkspaceAnalysis {
        min_reach,
        max_reach,
        samples: points.len(),
        points,
    }
}

fn generate_angle_combinations(
    limits: &[(f64, f64)],
    samples: usize,
    results: &mut Vec<Vec<f64>>,
    current: &mut Vec<f64>,
) {
    if current.len() == limits.len() {
        results.push(current.clone());
        return;
    }
    let idx = current.len();
    let (lo, hi) = limits[idx];
    for i in 0..samples {
        let t = i as f64 / (samples - 1).max(1) as f64;
        current.push(lo + t * (hi - lo));
        generate_angle_combinations(limits, samples, results, current);
        current.pop();
    }
}

/// Check if a target point is reachable.
pub fn is_reachable(
    base_params: &DhParams,
    angle_limits: &[(f64, f64)],
    target: (f64, f64, f64),
    samples_per_joint: usize,
    tolerance: f64,
) -> bool {
    let ws = analyze_workspace(base_params, angle_limits, samples_per_joint);
    let dist = ((target.0).powi(2) + (target.1).powi(2) + (target.2).powi(2)).sqrt();
    // Quick check: within min/max reach
    if dist > ws.max_reach + tolerance || dist < ws.min_reach - tolerance {
        return false;
    }
    // Fine check: any sample point close enough
    for &(x, y, z) in &ws.points {
        let d = ((x - target.0).powi(2) + (y - target.1).powi(2) + (z - target.2).powi(2)).sqrt();
        if d < tolerance {
            return true;
        }
    }
    false
}

/// Compute the workspace volume (approximate) by counting reachable voxels.
pub fn workspace_volume(
    base_params: &DhParams,
    angle_limits: &[(f64, f64)],
    samples_per_joint: usize,
    voxel_size: f64,
) -> f64 {
    let ws = analyze_workspace(base_params, angle_limits, samples_per_joint);
    // Count unique voxels
    let mut voxels = std::collections::HashSet::new();
    for &(x, y, z) in &ws.points {
        let vx = (x / voxel_size).round() as i64;
        let vy = (y / voxel_size).round() as i64;
        let vz = (z / voxel_size).round() as i64;
        voxels.insert((vx, vy, vz));
    }
    voxels.len() as f64 * voxel_size.powi(3)
}

/// Compute reachability map for a 2D cross-section.
pub fn reachability_map_2d(
    base_params: &DhParams,
    angle_limits: &[(f64, f64)],
    samples_per_joint: usize,
    grid_size: usize,
    bounds: (f64, f64, f64, f64), // (min_x, min_y, max_x, max_y)
    z_plane: f64,
) -> Vec<Vec<bool>> {
    let ws = analyze_workspace(base_params, angle_limits, samples_per_joint);
    let mut grid = vec![vec![false; grid_size]; grid_size];

    let dx = (bounds.2 - bounds.0) / grid_size as f64;
    let dy = (bounds.3 - bounds.1) / grid_size as f64;

    for &(px, py, pz) in &ws.points {
        if (pz - z_plane).abs() > dx * 2.0 { continue; }
        let gx = ((px - bounds.0) / dx) as usize;
        let gy = ((py - bounds.1) / dy) as usize;
        if gx < grid_size && gy < grid_size {
            grid[gy][gx] = true;
        }
    }
    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::two_link_arm;

    #[test]
    fn test_workspace_2link() {
        let arm = two_link_arm(1.0, 1.0, 0.0, 0.0);
        let limits = vec![(-std::f64::consts::PI, std::f64::consts::PI), (-std::f64::consts::PI, std::f64::consts::PI)];
        let ws = analyze_workspace(&arm, &limits, 20);
        assert!(ws.max_reach > 1.9, "max_reach={}", ws.max_reach);
        assert!(ws.min_reach < 0.5);
        assert!(ws.samples > 0);
    }

    #[test]
    fn test_reachability() {
        let arm = two_link_arm(1.0, 1.0, 0.0, 0.0);
        let limits = vec![(-std::f64::consts::PI, std::f64::consts::PI), (-std::f64::consts::PI, std::f64::consts::PI)];
        // (1.5, 0, 0) should be reachable
        assert!(is_reachable(&arm, &limits, (1.5, 0.0, 0.0), 20, 0.2));
        // (5, 0, 0) should not be reachable
        assert!(!is_reachable(&arm, &limits, (5.0, 0.0, 0.0), 20, 0.2));
    }

    #[test]
    fn test_workspace_min_max() {
        // For a 2-link arm with equal links, max reach = l1 + l2, min reach = 0
        let arm = two_link_arm(1.0, 1.0, 0.0, 0.0);
        let limits = vec![(-std::f64::consts::PI, std::f64::consts::PI), (-std::f64::consts::PI, std::f64::consts::PI)];
        let ws = analyze_workspace(&arm, &limits, 30);
        assert!((ws.max_reach - 2.0).abs() < 0.1, "max_reach={}", ws.max_reach);
    }

    #[test]
    fn test_workspace_volume_positive() {
        let arm = two_link_arm(1.0, 1.0, 0.0, 0.0);
        let limits = vec![(0.0, std::f64::consts::PI), (0.0, std::f64::consts::PI)];
        let vol = workspace_volume(&arm, &limits, 15, 0.1);
        assert!(vol > 0.0);
    }

    #[test]
    fn test_reachability_map_2d() {
        let arm = two_link_arm(1.0, 1.0, 0.0, 0.0);
        let limits = vec![(-std::f64::consts::PI, std::f64::consts::PI), (-std::f64::consts::PI, std::f64::consts::PI)];
        let map = reachability_map_2d(&arm, &limits, 20, 20, (-3.0, -3.0, 3.0, 3.0), 0.0);
        // Center should be reachable
        let cx = 10;
        let cy = 10;
        assert!(map[cy][cx] || map[cy+1][cx] || map[cy][cx+1]);
    }
}
