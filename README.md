# kinematics

Robot kinematics library: forward kinematics using Denavit-Hartenberg parameters, inverse kinematics via Jacobian methods, workspace analysis. Pure Rust, no external dependencies.

## Features

- **DH Parameters**: Standard DH convention with 4x4 homogeneous transforms
- **Forward Kinematics**: Compute end-effector pose from joint angles
- **Inverse Kinematics**: Damped least-squares Jacobian method, analytical 2-link IK
- **Jacobian**: Numerical and analytical computation, manipulability, singularity detection
- **Workspace Analysis**: Reachability, volume estimation, 2D cross-sections

## License

MIT
