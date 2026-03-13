//! Example: run Relaxed IK for multiple kinematic chains with custom objectives,
//! including shared-joint handling.
//!
//! Shows how to:
//! 1. Define custom objectives (implementing `ObjectiveTrait`)
//! 2. Detect shared joints between kinematic chains
//! 3. Use approach 2.1 (penalty) or 2.2 (variable reduction) for shared joints
//! 4. Validate shared joint consistency after solving
//!
//! Run from the project root:
//!   cargo run --example multi_chain_ik

extern crate relaxed_ik_lib;
use relaxed_ik_lib::relaxed_ik::{self, SharedJointMode};
use relaxed_ik_lib::utils_rust::file_utils::get_path_to_src;
use relaxed_ik_lib::groove::objective::{ObjectiveTrait, groove_loss};
use relaxed_ik_lib::groove::vars::RelaxedIKVars;
use nalgebra::{Vector3, UnitQuaternion, Quaternion};

// ----- Custom objectives (implement ObjectiveTrait) -----

struct PreferJointValue {
    joint_idx: usize,
    target: f64,
}
impl PreferJointValue {
    fn new(joint_idx: usize, target: f64) -> Self {
        Self { joint_idx, target }
    }
}
impl ObjectiveTrait for PreferJointValue {
    fn call(
        &self,
        x: &[f64],
        _v: &RelaxedIKVars,
        _frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>,
    ) -> f64 {
        let val = x[self.joint_idx] - self.target;
        groove_loss(val, 0.0, 2, 0.15, 10.0, 2)
    }
    fn call_lite(
        &self,
        x: &[f64],
        _v: &RelaxedIKVars,
        _ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>,
    ) -> f64 {
        let val = x[self.joint_idx] - self.target;
        groove_loss(val, 0.0, 2, 0.15, 10.0, 2)
    }
}

struct PreferJointNearZero {
    joint_idx: usize,
}
impl PreferJointNearZero {
    fn new(joint_idx: usize) -> Self {
        Self { joint_idx }
    }
}
impl ObjectiveTrait for PreferJointNearZero {
    fn call(
        &self,
        x: &[f64],
        _v: &RelaxedIKVars,
        _frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>,
    ) -> f64 {
        let val = x[self.joint_idx];
        groove_loss(val, 0.0, 2, 0.2, 5.0, 2)
    }
    fn call_lite(
        &self,
        x: &[f64],
        _v: &RelaxedIKVars,
        _ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>,
    ) -> f64 {
        let val = x[self.joint_idx];
        groove_loss(val, 0.0, 2, 0.2, 5.0, 2)
    }
}

fn main() {
    let path_to_src = get_path_to_src();
    let settings_path = path_to_src + "configs/settings.yaml";
    let mut relaxed_ik = relaxed_ik::RelaxedIK::load_settings(settings_path.as_str());

    let num_chains = relaxed_ik.vars.robot.num_chains;
    let num_dofs = relaxed_ik.vars.xopt.len();
    println!("Running IK for {} kinematic chain(s), {} DoF total.", num_chains, num_dofs);

    // --- Shared joint handling ---
    if relaxed_ik.vars.robot.has_shared_joints() {
        println!("\nShared joints detected! Choosing a handling mode...");

        // Approach 2.1: Penalty-based (add alignment cost terms with heavy weight).
        // Uncomment ONE of the two approaches below.
        relaxed_ik.enable_shared_joint_penalty(200.0);

        // Approach 2.2: Variable reduction (single variable per physical joint).
        // relaxed_ik.enable_shared_joint_reduction();
    } else {
        println!("\nNo shared joints detected between chains.");
    }

    // Add custom objectives
    relaxed_ik.om.objectives.push(Box::new(PreferJointNearZero::new(0)));
    relaxed_ik.om.weight_priors.push(0.4);
    relaxed_ik.om.objectives.push(Box::new(PreferJointValue::new(1, 0.5)));
    relaxed_ik.om.weight_priors.push(0.3);

    // Example: run 10 steps, moving each end-effector goal along the y-axis.
    for step in 0..10 {
        for chain in 0..num_chains {
            relaxed_ik.vars.goal_positions[chain] += Vector3::new(0.0, 0.01, 0.0);
        }
        let joint_solution = relaxed_ik.solve();
        println!("Step {} — joint solution: {:?}", step, joint_solution);

        // Validate shared joints after each solve
        if relaxed_ik.vars.robot.has_shared_joints() {
            relaxed_ik.print_shared_joint_validation(&joint_solution);
        }
    }
}
