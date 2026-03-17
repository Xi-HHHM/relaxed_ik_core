use crate::groove::vars::RelaxedIKVars;
use crate::groove::groove::{OptimizationEngineOpen};
use crate::groove::objective_master::{ObjectiveMaster, ObjectiveWeightsConfig};
pub use crate::groove::objective_master::ObjectiveReportMode;
use crate::groove::objective::{SharedJointAlignment, RelativeTCPConstraint};
use crate::utils_rust::file_utils::{*};
use std::fs::File;
use std::io::Read;
use yaml_rust::YamlLoader;
use crate::utils_rust::transformations::{*};
use nalgebra::{Vector3, UnitQuaternion, Quaternion};
use std::os::raw::{c_double, c_int};

#[repr(C)]
pub struct Opt {
    pub data: *const c_double,
    pub length: c_int,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SharedJointMode {
    /// No special handling (default, or when no shared joints exist).
    None,
    /// Approach 2.1: Add heavily-weighted penalty terms to align shared joints.
    Penalty,
    /// Approach 2.2: Optimize over reduced variables (one variable per physical joint).
    VariableReduction,
}

pub struct RelaxedIK {
    pub vars: RelaxedIKVars,
    pub om: ObjectiveMaster,
    pub groove: OptimizationEngineOpen,
    max_iterations: usize,
    pub shared_joint_mode: SharedJointMode,
    groove_reduced: Option<OptimizationEngineOpen>,
    report_mode: ObjectiveReportMode,
}

impl RelaxedIK {
    pub fn load_settings(path_to_setting: &str) -> Self {
        println!("RelaxedIK is using below setting file {}", path_to_setting);

        let vars = RelaxedIKVars::from_local_settings(path_to_setting);

        let weights = {
            let mut contents = String::new();
            if let Ok(mut f) = File::open(path_to_setting) {
                let _ = f.read_to_string(&mut contents);
                if let Ok(docs) = YamlLoader::load_from_str(&contents) {
                    if let Some(settings) = docs.first() {
                        ObjectiveWeightsConfig::from_yaml(settings)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        let mut om = ObjectiveMaster::relaxed_ik(&vars.robot.chain_lengths, weights.as_ref());

        let rel_tcp_enabled = {
            let mut contents = String::new();
            if let Ok(mut f) = File::open(path_to_setting) {
                let _ = f.read_to_string(&mut contents);
                if let Ok(docs) = YamlLoader::load_from_str(&contents) {
                    if let Some(settings) = docs.first() {
                        settings["relative_tcp_constraints"].as_bool().unwrap_or(false)
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        };

        let rel_tcp_weight = weights.as_ref()
            .map(|w| w.get("RelativeTCPConstraint"))
            .unwrap_or_else(|| ObjectiveWeightsConfig::default().get("RelativeTCPConstraint"));

        if rel_tcp_enabled && vars.robot.num_chains >= 2 {
            for a in 0..vars.robot.num_chains {
                for b in (a + 1)..vars.robot.num_chains {
                    om.objectives.push(Box::new(RelativeTCPConstraint::new(a, b)));
                    om.weight_priors.push(rel_tcp_weight);
                }
            }
            println!("RelativeTCPConstraint enabled (weight={}) for {} pair(s)",
                rel_tcp_weight, (vars.robot.num_chains * (vars.robot.num_chains - 1)) / 2);
        }

        let groove = OptimizationEngineOpen::new(vars.robot.num_dofs.clone());

        Self{vars, om, groove, max_iterations: 100, shared_joint_mode: SharedJointMode::None, groove_reduced: None, report_mode: ObjectiveReportMode::Off}
    }

    /// Approach 2.1: Add penalty objectives that force shared joint variables to match.
    /// `weight` is the penalty coefficient in `weight * (x[a] - x[b])^2`.
    /// Reasonable values: 1e4 – 1e7 (must dominate the position objectives
    /// which total ~600 effective weight across all chains).
    pub fn enable_shared_joint_penalty(&mut self, weight: f64) {
        assert!(self.vars.robot.has_shared_joints(),
            "No shared joints detected between chains — penalty mode is not needed.");
        self.shared_joint_mode = SharedJointMode::Penalty;
        for &(a, b) in &self.vars.robot.shared_joint_pairs {
            self.om.objectives.push(Box::new(SharedJointAlignment::new(a, b, weight)));
            self.om.weight_priors.push(1.0);
        }
        println!("SharedJointMode::Penalty enabled with weight={} for {} pair(s)",
            weight, self.vars.robot.shared_joint_pairs.len());
    }

    /// Enable relative TCP constraints: keep end-effectors in the relative pose implied by
    /// current goals. Useful when absolute goal tracking can be relaxed but formation
    /// (relative geometry between TCPs) must be maintained. Requires num_chains >= 2.
    /// `weight` defaults to 100.0 if not overridden via set_objective_weight.
    pub fn enable_relative_tcp_constraints(&mut self, weight: f64) {
        if self.vars.robot.num_chains < 2 {
            return;
        }

        // Idempotent behavior: if constraints were already added (e.g. via YAML load
        // or a previous call), just update their weights instead of duplicating terms.
        let already_added = self.om.objectives.iter().any(|o| {
            o.name().split('[').next().unwrap_or("").trim() == "RelativeTCPConstraint"
        });
        if already_added {
            self.om.set_objective_weight("RelativeTCPConstraint", weight);
            println!("RelativeTCPConstraint updated (weight={})", weight);
            return;
        }

        for a in 0..self.vars.robot.num_chains {
            for b in (a + 1)..self.vars.robot.num_chains {
                self.om.objectives.push(Box::new(RelativeTCPConstraint::new(a, b)));
                self.om.weight_priors.push(weight);
            }
        }
        println!("RelativeTCPConstraint enabled (weight={}) for {} pair(s)",
            weight, (self.vars.robot.num_chains * (self.vars.robot.num_chains - 1)) / 2);
    }

    /// Approach 2.2: Optimize in reduced variable space (one variable per physical joint).
    /// Gradients from both chains are aggregated into shared variables.
    pub fn enable_shared_joint_reduction(&mut self) {
        assert!(self.vars.robot.has_shared_joints(),
            "No shared joints detected between chains — reduction mode is not needed.");
        self.shared_joint_mode = SharedJointMode::VariableReduction;
        self.groove_reduced = Some(OptimizationEngineOpen::new(self.vars.robot.num_unique_dofs));
        println!("SharedJointMode::VariableReduction enabled: {} full DOFs -> {} unique DOFs",
            self.vars.robot.num_dofs, self.vars.robot.num_unique_dofs);
    }

    /// Set objective report verbosity: Off (default), Brief (one row per class), Detailed.
    pub fn set_objective_report_mode(&mut self, mode: ObjectiveReportMode) {
        self.report_mode = mode;
    }

    /// Set optimizer iteration budget per solve call. Must be >= 1.
    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        assert!(max_iterations >= 1, "max_iterations must be >= 1");
        self.max_iterations = max_iterations;
    }

    /// Get current optimizer iteration budget.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Set weight for all objectives of a given class. Class names:
    /// MatchEEPosiDoF, MatchEERotaDoF, EachJointLimits, MinimizeVelocity,
    /// MinimizeAcceleration, MinimizeJerk, MaximizeManipulability, SelfCollision.
    pub fn set_objective_weight(&mut self, class: &str, weight: f64) {
        self.om.set_objective_weight(class, weight);
    }

    pub fn reset(&mut self, x: Vec<f64>) {
        self.vars.reset( x.clone());
    }

    pub fn solve(&mut self) -> Vec<f64> {
        match self.shared_joint_mode {
            SharedJointMode::None | SharedJointMode::Penalty => {
                self.solve_full_space()
            },
            SharedJointMode::VariableReduction => {
                self.solve_reduced_space()
            },
        }
    }

    fn solve_full_space(&mut self) -> Vec<f64> {
        let mut out_x = self.vars.xopt.clone();

        self.groove.optimize(&mut out_x, &self.vars, &self.om, self.max_iterations);

        for i in 0..out_x.len() {
            if out_x[i].is_nan() {
                println!("No valid solution found! Returning previous solution: {:?}. End effector position goals: {:?}", self.vars.xopt, self.vars.goal_positions);
                return self.vars.xopt.clone();
            }
        }
        self.vars.update(out_x.clone());
        self.om.print_objective_report(&out_x, &self.vars, self.report_mode, self.max_iterations);
        out_x
    }

    fn solve_reduced_space(&mut self) -> Vec<f64> {
        let mut x_reduced = self.vars.robot.reduce_to_unique(&self.vars.xopt);

        if let Some(ref mut groove_r) = self.groove_reduced {
            groove_r.optimize_reduced(&mut x_reduced, &self.vars, &self.om, self.max_iterations);
        }

        let out_x = self.vars.robot.expand_to_full(&x_reduced);

        for i in 0..out_x.len() {
            if out_x[i].is_nan() {
                println!("No valid solution found! Returning previous solution: {:?}. End effector position goals: {:?}", self.vars.xopt, self.vars.goal_positions);
                return self.vars.xopt.clone();
            }
        }
        self.vars.update(out_x.clone());
        self.om.print_objective_report(&out_x, &self.vars, self.report_mode, self.max_iterations);
        out_x
    }

    /// Validate shared joints: returns a list of (joint_name, idx_a, idx_b, abs_difference).
    /// For Penalty mode, nonzero differences indicate the penalty wasn't strong enough.
    /// For VariableReduction mode, differences should always be exactly 0.
    pub fn validate_shared_joints(&self, x: &[f64]) -> Vec<(String, usize, usize, f64)> {
        let mut results = Vec::new();
        for &(a, b) in &self.vars.robot.shared_joint_pairs {
            let name = self.vars.robot.all_joint_names[a].clone();
            let diff = (x[a] - x[b]).abs();
            results.push((name, a, b, diff));
        }
        results
    }

    /// Print a summary of shared joint validation.
    pub fn print_shared_joint_validation(&self, x: &[f64]) {
        let results = self.validate_shared_joints(x);
        if results.is_empty() {
            println!("No shared joints to validate.");
            return;
        }
        let max_diff = results.iter().map(|r| r.3).fold(0.0f64, f64::max);
        println!("Shared joint validation (mode={:?}):", self.shared_joint_mode);
        for (name, a, b, diff) in &results {
            let status = if *diff < 1e-6 { "OK" } else { "MISMATCH" };
            println!("  [{}] {} : x[{}]={:.6} vs x[{}]={:.6}, diff={:.2e}",
                status, name, a, x[*a], b, x[*b], diff);
        }
        println!("  Max difference: {:.2e}", max_diff);
    }
}
