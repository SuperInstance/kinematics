//! # kinematics
//!
//! Robot kinematics: forward kinematics using Denavit-Hartenberg parameters,
//! inverse kinematics via Jacobian methods, workspace analysis.
//! Pure Rust, no external dependencies.

#![allow(clippy::needless_range_loop, clippy::ptr_arg)]

pub mod dh_param;
pub mod forward;
pub mod inverse;
pub mod jacobian;
pub mod workspace;

pub use dh_param::DhParams;
pub use forward::forward_kinematics;
pub use inverse::inverse_kinematics;
pub use jacobian::Jacobian;
