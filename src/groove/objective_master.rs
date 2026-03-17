use crate::groove::objective::*;
use crate::groove::vars::RelaxedIKVars;
use std::collections::HashMap;

/// Parse "RelativeTCPConstraint[a-b]" -> Some((a, b)) or None.
fn parse_relative_tcp_indices(name: &str) -> Option<(usize, usize)> {
    let prefix = "RelativeTCPConstraint[";
    if !name.starts_with(prefix) {
        return None;
    }
    let rest = &name[prefix.len()..];
    let end = rest.find(']')?;
    let middle = rest[..end].find('-')?;
    let a: usize = rest[..middle].parse().ok()?;
    let b: usize = rest[middle + 1..end].parse().ok()?;
    Some((a, b))
}

/// Controls verbosity of objective reports after each solve.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObjectiveReportMode {
    /// No objective reports (default).
    Off,
    /// One row per objective class (e.g., MatchEEPosiDoF, EachJointLimits).
    Brief,
    /// One row per objective instance (detailed).
    Detailed,
}

/// Extract class name from objective name, e.g. "MatchEEPosiDoF[arm=0,axis=0]" -> "MatchEEPosiDoF".
fn objective_class(name: &str) -> &str {
    name.split('[').next().unwrap_or(name).trim()
}

/// Default weights for relaxed_ik objectives. Used when not overridden by YAML or API.
pub const DEFAULT_MATCH_EE_POSI_DOF: f64 = 50.0;
pub const DEFAULT_MATCH_EE_ROTA_DOF: f64 = 10.0;
pub const DEFAULT_EACH_JOINT_LIMITS: f64 = 0.1;
pub const DEFAULT_MINIMIZE_VELOCITY: f64 = 0.7;
pub const DEFAULT_MINIMIZE_ACCELERATION: f64 = 0.5;
pub const DEFAULT_MINIMIZE_JERK: f64 = 0.3;
pub const DEFAULT_MAXIMIZE_MANIPULABILITY: f64 = 1.0;
pub const DEFAULT_SELF_COLLISION: f64 = 0.01;
pub const DEFAULT_RELATIVE_TCP_CONSTRAINTS: f64 = 0.0;

/// Configurable objective weights. Keys: MatchEEPosiDoF, MatchEERotaDoF, EachJointLimits,
/// MinimizeVelocity, MinimizeAcceleration, MinimizeJerk, MaximizeManipulability, SelfCollision.
#[derive(Clone, Default)]
pub struct ObjectiveWeightsConfig {
    pub overrides: HashMap<String, f64>,
}

impl ObjectiveWeightsConfig {
    pub fn new() -> Self { Self::default() }

    pub fn set(&mut self, class: &str, weight: f64) {
        self.overrides.insert(class.to_string(), weight);
    }

    pub fn get(&self, class: &str) -> f64 {
        self.overrides.get(class).copied().unwrap_or_else(|| match class {
            "MatchEEPosiDoF" => DEFAULT_MATCH_EE_POSI_DOF,
            "MatchEERotaDoF" => DEFAULT_MATCH_EE_ROTA_DOF,
            "EachJointLimits" => DEFAULT_EACH_JOINT_LIMITS,
            "MinimizeVelocity" => DEFAULT_MINIMIZE_VELOCITY,
            "MinimizeAcceleration" => DEFAULT_MINIMIZE_ACCELERATION,
            "MinimizeJerk" => DEFAULT_MINIMIZE_JERK,
            "MaximizeManipulability" => DEFAULT_MAXIMIZE_MANIPULABILITY,
            "SelfCollision" => DEFAULT_SELF_COLLISION,
            "RelativeTCPConstraint" => DEFAULT_RELATIVE_TCP_CONSTRAINTS,
            _ => 0.1, // fallback for unknown
        })
    }

    /// Parse from YAML settings. Expects optional "objective_weights" map with snake_case keys.
    pub fn from_yaml(settings: &yaml_rust::Yaml) -> Option<Self> {
        let w = settings["objective_weights"].as_hash()?;
        let mut overrides = HashMap::new();
        for (k, v) in w {
            let key = k.as_str()?;
            let val = v.as_f64()?;
            let class = match key {
                "match_ee_posi_dof" => "MatchEEPosiDoF",
                "match_ee_rota_dof" => "MatchEERotaDoF",
                "each_joint_limits" => "EachJointLimits",
                "minimize_velocity" => "MinimizeVelocity",
                "minimize_acceleration" => "MinimizeAcceleration",
                "minimize_jerk" => "MinimizeJerk",
                "maximize_manipulability" => "MaximizeManipulability",
                "self_collision" => "SelfCollision",
                "relative_tcp_constraints" => "RelativeTCPConstraint",
                _ => continue,
            };
            overrides.insert(class.to_string(), val);
        }
        if overrides.is_empty() { None } else { Some(Self { overrides }) }
    }
}

pub struct ObjectiveMaster {
    pub objectives: Vec<Box<dyn ObjectiveTrait + Send>>,
    pub num_chains: usize,
    pub weight_priors: Vec<f64>,
    pub lite: bool,
    pub finite_diff_grad: bool
}

impl ObjectiveMaster {
    pub fn standard_ik(num_chains: usize) -> Self {
        let mut objectives: Vec<Box<dyn ObjectiveTrait + Send>> = Vec::new();
        let mut weight_priors: Vec<f64> = Vec::new();
        for i in 0..num_chains {
            objectives.push(Box::new(MatchEEPosGoals::new(i)));
            weight_priors.push(1.0);
            objectives.push(Box::new(MatchEEQuatGoals::new(i)));
            weight_priors.push(1.0);
        }
        Self{objectives, num_chains, weight_priors, lite: true, finite_diff_grad: true}
    }


    pub fn relaxed_ik(chain_lengths: &[usize], weights: Option<&ObjectiveWeightsConfig>) -> Self {
        let get_w = |class: &str| weights.map_or_else(
            || ObjectiveWeightsConfig::default().get(class),
            |c| c.get(class)
        );

        let mut objectives: Vec<Box<dyn ObjectiveTrait + Send>> = Vec::new();
        let mut weight_priors: Vec<f64> = Vec::new();
        let num_chains = chain_lengths.len();
        let mut num_dofs = 0;
        for i in 0..num_chains {
            let wp = get_w("MatchEEPosiDoF");
            objectives.push(Box::new(MatchEEPosiDoF::new(i, 0)));
            weight_priors.push(wp);
            objectives.push(Box::new(MatchEEPosiDoF::new(i, 1)));
            weight_priors.push(wp);
            objectives.push(Box::new(MatchEEPosiDoF::new(i, 2)));
            weight_priors.push(wp);
            let wr = get_w("MatchEERotaDoF");
            objectives.push(Box::new(MatchEERotaDoF::new(i, 0)));
            weight_priors.push(wr);
            objectives.push(Box::new(MatchEERotaDoF::new(i, 1)));
            weight_priors.push(wr);
            objectives.push(Box::new(MatchEERotaDoF::new(i, 2)));
            weight_priors.push(wr);
            num_dofs += chain_lengths[i];
        }

        let wj = get_w("EachJointLimits");
        for j in 0..num_dofs {
            objectives.push(Box::new(EachJointLimits::new(j)));
            weight_priors.push(wj);
        }

        objectives.push(Box::new(MinimizeVelocity));
        weight_priors.push(get_w("MinimizeVelocity"));
        objectives.push(Box::new(MinimizeAcceleration));
        weight_priors.push(get_w("MinimizeAcceleration"));
        objectives.push(Box::new(MinimizeJerk));
        weight_priors.push(get_w("MinimizeJerk"));
        objectives.push(Box::new(MaximizeManipulability));
        weight_priors.push(get_w("MaximizeManipulability"));

        let wsc = get_w("SelfCollision");
        for i in 0..num_chains {
            for j in 0..chain_lengths[i]-2 {
                for k in j+2..chain_lengths[i] {
                    objectives.push(Box::new(SelfCollision::new(0, j, k)));
                    weight_priors.push(wsc);
                }
            }
        }

        Self{objectives, num_chains, weight_priors, lite: false, finite_diff_grad: false}
    }

    /// Set weight for all objectives of a given class (e.g. "MinimizeVelocity", "SelfCollision").
    pub fn set_objective_weight(&mut self, class: &str, weight: f64) {
        for i in 0..self.objectives.len() {
            if objective_class(&self.objectives[i].name()) == class {
                self.weight_priors[i] = weight;
            }
        }
    }

    pub fn call(&self, x: &[f64], vars: &RelaxedIKVars) -> f64 {
        if self.lite {
            self.__call_lite(x, vars)
        } else {
            self.__call(x, vars)
        }
    }

    pub fn gradient(&self, x: &[f64], vars: &RelaxedIKVars) -> (f64, Vec<f64>) {
        if self.lite {
            if self.finite_diff_grad {
                self.__gradient_finite_diff_lite(x, vars)
            } else {
                self.__gradient_lite(x, vars)
            }
        } else {
            if self.finite_diff_grad {
                self.__gradient_finite_diff(x, vars)
            } else {
                self.__gradient(x, vars)
            }
        }
    }

    pub fn gradient_finite_diff(&self, x: &[f64], vars: &RelaxedIKVars) -> (f64, Vec<f64>) {
        if self.lite {
            self.__gradient_finite_diff_lite(x, vars)
        } else {
            self.__gradient_finite_diff(x, vars)
        }
    }

    fn __call(&self, x: &[f64], vars: &RelaxedIKVars) -> f64 {
        let mut out = 0.0;
        let frames = vars.robot.get_frames_immutable(x);
        for i in 0..self.objectives.len() {
            out += self.weight_priors[i] * self.objectives[i].call(x, vars, &frames);
        }
        out
    }

    fn __call_lite(&self, x: &[f64], vars: &RelaxedIKVars) -> f64 {
        let mut out = 0.0;
        let poses = vars.robot.get_ee_pos_and_quat_immutable(x);
        for i in 0..self.objectives.len() {
            out += self.weight_priors[i] * self.objectives[i].call_lite(x, vars, &poses);
        }
        out
    }

    fn __gradient(&self, x: &[f64], vars: &RelaxedIKVars) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = vec![0. ; x.len()];
        let mut obj = 0.0;

        let mut finite_diff_list: Vec<usize> = Vec::new();
        let mut f_0s: Vec<f64> = Vec::new();
        let frames_0 = vars.robot.get_frames_immutable(x);
        for i in 0..self.objectives.len() {
            if self.objectives[i].gradient_type() == 0 {
                let (local_obj, local_grad) = self.objectives[i].gradient(x, vars, &frames_0);
                f_0s.push(local_obj);
                obj += self.weight_priors[i] * local_obj;
                for j in 0..local_grad.len() {
                    grad[j] += self.weight_priors[i] * local_grad[j];
                }
            } else if self.objectives[i].gradient_type() == 1 {
                finite_diff_list.push(i);
                let local_obj = self.objectives[i].call(x, vars, &frames_0);
                obj += self.weight_priors[i] * local_obj;
                f_0s.push(local_obj);
            }
        }

        if finite_diff_list.len() > 0 {
            for i in 0..x.len() {
                let mut x_h = x.to_vec();
                x_h[i] += 0.0000001;
                let frames_h = vars.robot.get_frames_immutable(x_h.as_slice());
                for j in &finite_diff_list {
                    let f_h = self.objectives[*j].call(&x_h, vars, &frames_h);
                    grad[i] += self.weight_priors[*j] * ((-f_0s[*j] + f_h) /  0.0000001);
                }
            }
        }

        (obj, grad)
    }

    fn __gradient_lite(&self, x: &[f64], vars: &RelaxedIKVars) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = vec![0. ; x.len()];
        let mut obj = 0.0;

        let mut finite_diff_list: Vec<usize> = Vec::new();
        let mut f_0s: Vec<f64> = Vec::new();
        let poses_0 = vars.robot.get_ee_pos_and_quat_immutable(x);
        for i in 0..self.objectives.len() {
            if self.objectives[i].gradient_type() == 1 {
                let (local_obj, local_grad) = self.objectives[i].gradient_lite(x, vars, &poses_0);
                f_0s.push(local_obj);
                obj += self.weight_priors[i] * local_obj;
                for j in 0..local_grad.len() {
                    grad[j] += self.weight_priors[i] * local_grad[j];
                }
            } else if self.objectives[i].gradient_type() == 0 {
                finite_diff_list.push(i);
                let local_obj = self.objectives[i].call_lite(x, vars, &poses_0);
                obj += self.weight_priors[i] * local_obj;
                f_0s.push(local_obj);
            }
        }

        if finite_diff_list.len() > 0 {
            for i in 0..x.len() {
                let mut x_h = x.to_vec();
                x_h[i] += 0.0000001;
                let poses_h = vars.robot.get_ee_pos_and_quat_immutable(x_h.as_slice());
                for j in &finite_diff_list {
                    let f_h = self.objectives[*j].call_lite(x, vars, &poses_h);
                    grad[i] += self.weight_priors[*j] * ((-f_0s[*j] + f_h) /  0.0000001);
                }
            }
        }

        (obj, grad)
    }

    fn __gradient_finite_diff(&self, x: &[f64], vars: &RelaxedIKVars) -> (f64, Vec<f64>)  {
        let mut grad: Vec<f64> = vec![0. ; x.len()];
        let mut f_0 = self.call(x, vars);

        for i in 0..x.len() {
            let mut x_h = x.to_vec();
            x_h[i] += 0.000001;
            let f_h = self.call(x_h.as_slice(), vars);
            grad[i] = (-f_0 + f_h) / 0.000001;
        }

        (f_0, grad)
    }

    fn __gradient_finite_diff_lite(&self, x: &[f64], vars: &RelaxedIKVars) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = vec![0. ; x.len()];
        let mut f_0 = self.call(x, vars);

        for i in 0..x.len() {
            let mut x_h = x.to_vec();
            x_h[i] += 0.000001;
            let f_h = self.__call_lite(x_h.as_slice(), vars);
            grad[i] = (-f_0 + f_h) / 0.000001;
        }

        (f_0, grad)
    }

    /// Print objectives, weights, and residuals based on the given report mode.
    pub fn print_objective_report(&self, x: &[f64], vars: &RelaxedIKVars, mode: ObjectiveReportMode, max_iterations: usize) {
        if mode == ObjectiveReportMode::Off {
            return;
        }

        let mut total: f64 = 0.0;
        let rows: Vec<(String, f64, f64, f64)> = if self.lite {
            let poses = vars.robot.get_ee_pos_and_quat_immutable(x);
            (0..self.objectives.len()).map(|i| {
                let residual = self.objectives[i].call_lite(x, vars, &poses);
                let weighted = self.weight_priors[i] * residual;
                total += weighted;
                (self.objectives[i].name(), self.weight_priors[i], residual, weighted)
            }).collect()
        } else {
            let frames = vars.robot.get_frames_immutable(x);
            (0..self.objectives.len()).map(|i| {
                let residual = self.objectives[i].call(x, vars, &frames);
                let weighted = self.weight_priors[i] * residual;
                total += weighted;
                (self.objectives[i].name(), self.weight_priors[i], residual, weighted)
            }).collect()
        };

        match mode {
            ObjectiveReportMode::Off => {}
            ObjectiveReportMode::Brief => {
                // Group by class: (count, sum_residual, sum_weighted, min_res, max_res)
                let mut by_class: HashMap<String, (usize, f64, f64, Option<f64>, Option<f64>)> = HashMap::new();
                for (name, _w, res, weighted) in &rows {
                    let class = objective_class(name).to_string();
                    let entry = by_class.entry(class).or_insert((0, 0.0, 0.0, None, None));
                    entry.0 += 1;
                    entry.1 += res;
                    entry.2 += weighted;
                    entry.3 = Some(entry.3.map_or(*res, |m| m.min(*res)));
                    entry.4 = Some(entry.4.map_or(*res, |m| m.max(*res)));
                }
                let mut classes: Vec<_> = by_class.keys().collect();
                classes.sort();
                println!("--- Objective Report (brief, x_len={}, max_iterations={}) ---", x.len(), max_iterations);
                for class in classes {
                    let (count, sum_res, sum_weighted, min_res, max_res) = &by_class[class];
                    let mean_res = sum_res / *count as f64;
                    let (min_s, max_s) = match (min_res, max_res) {
                        (Some(a), Some(b)) => (format!("{:12.3e}", a), format!("{:12.3e}", b)),
                        _ => ("N/A".into(), "N/A".into()),
                    };
                    println!("  {:30}  n={:3}  mean_res={:12.3e}  [min,max]=[{},{}]  weighted_sum={:12.3e}",
                        class, count, mean_res, min_s, max_s, sum_weighted);
                }
                println!("  Total cost: {:12.3e}", total);
                println!("  (lower is better: negative = near goals, positive = constraint violations or large tracking errors)");
                println!("---");
            }
            ObjectiveReportMode::Detailed => {
                println!("--- Objective Report (detailed, x_len={}, max_iterations={}) ---", x.len(), max_iterations);
                let ee_poses = vars.robot.get_ee_pos_and_quat_immutable(x);
                for (i, (name, w, res, weighted)) in rows.iter().enumerate() {
                    println!("  {:3}: {:45}  weight={:8.4}  residual={:12.3e}  (weighted={:12.3e})",
                        i, name, w, res, weighted);
                    if let Some((a, b)) = parse_relative_tcp_indices(name) {
                        if a < vars.goal_positions.len() && b < vars.goal_positions.len()
                            && a < ee_poses.len() && b < ee_poses.len() {
                            let desired_xyz = vars.goal_positions[b] - vars.goal_positions[a];
                            let actual_xyz = ee_poses[b].0 - ee_poses[a].0;
                            let desired_quat = vars.goal_quats[a].inverse() * vars.goal_quats[b];
                            let actual_quat = ee_poses[a].1.inverse() * ee_poses[b].1;
                            let q = desired_quat.quaternion();
                            let qa = actual_quat.quaternion();
                            println!("       desired: xyz=[{:.6}, {:.6}, {:.6}]  quat_xyzw=[{:.6}, {:.6}, {:.6}, {:.6}]",
                                desired_xyz.x, desired_xyz.y, desired_xyz.z,
                                q.i, q.j, q.k, q.w);
                            println!("       actual:  xyz=[{:.6}, {:.6}, {:.6}]  quat_xyzw=[{:.6}, {:.6}, {:.6}, {:.6}]",
                                actual_xyz.x, actual_xyz.y, actual_xyz.z,
                                qa.i, qa.j, qa.k, qa.w);
                        }
                    }
                }
                println!("  Total cost: {:12.3e}", total);
                println!("  (lower is better: negative = near goals, positive = constraint violations or large tracking errors)");
                println!("---");
            }
        }
    }
}