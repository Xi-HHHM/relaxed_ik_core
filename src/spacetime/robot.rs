use crate::spacetime::arm;
use nalgebra;
use urdf_rs;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Robot {
    pub arms: Vec<arm::Arm>,
    pub num_chains: usize,
    pub num_dofs: usize,
    pub chain_lengths: Vec<usize>,
    pub lower_joint_limits: Vec<f64>,
    pub upper_joint_limits: Vec<f64>,
    pub all_joint_names: Vec<String>,
    pub joint_names_per_chain: Vec<Vec<String>>,
    /// Pairs of global DOF indices that refer to the same physical joint.
    pub shared_joint_pairs: Vec<(usize, usize)>,
    /// Maps each global DOF index to its unique DOF index (shared joints map to the same unique index).
    pub full_to_reduced: Vec<usize>,
    pub num_unique_dofs: usize,
    pub reduced_lower_limits: Vec<f64>,
    pub reduced_upper_limits: Vec<f64>,
}

impl Robot {
    pub fn from_urdf(urdf: &str, base_links: &[String], ee_links: &[String]) -> Self {
        
        // let chain = k::Chain::<f64>::from_urdf_file(urdf).unwrap();
        let description : urdf_rs::Robot = urdf_rs::read_from_string(urdf).unwrap();
        let chain: k::Chain<f64> = k::Chain::from(description.clone());

        let mut arms: Vec<arm::Arm> = Vec::new();
        let num_chains = base_links.len();
        let mut chain_lengths = Vec::new();
        let mut num_dofs = 0;

        let mut lower_joint_limits = Vec::new();
        let mut upper_joint_limits = Vec::new();
        let mut joint_names_per_chain: Vec<Vec<String>> = Vec::new();

        for i in 0..num_chains {
            let base_link = chain.find_link(base_links[i].as_str()).unwrap();
            let ee_link = chain.find_link(ee_links[i].as_str()).unwrap();
            let serial_chain = k::SerialChain::from_end_to_root(&ee_link, &base_link);

            let mut axis_types: Vec<String> = Vec::new();
            let mut joint_types: Vec<String> = Vec::new();
            let disp_offset = nalgebra::Vector3::new(0.0, 0.0, 0.0);
            let mut displacements = Vec::new();
            let mut rot_offsets = Vec::new();
            let mut chain_joint_names: Vec<String> = Vec::new();

            let mut first_link: bool = true;
            serial_chain.iter().for_each(|node| {
                let joint = node.joint();
                if first_link {
                    first_link = false;
                    return
                } else {
                    match joint.joint_type {
                        k::JointType::Fixed => {
                            joint_types.push("fixed".to_string());
                        },
                        k::JointType::Rotational { axis } => {
                            chain_joint_names.push(joint.name.clone());
                            if axis[0] == 1.0 {
                                axis_types.push("x".to_string());
                            } else if axis[1] == 1.0 {
                                axis_types.push("y".to_string());
                            } else if axis[2] == 1.0 {
                                axis_types.push("z".to_string());
                            } else if axis[0] == -1.0 {
                                axis_types.push("-x".to_string());
                            } else if axis[1] == -1.0 {
                                axis_types.push("-y".to_string());
                            } else if axis[2] == -1.0 {
                                axis_types.push("-z".to_string());
                            }
                            if joint.limits.is_none() {
                                joint_types.push("continuous".to_string());
                                lower_joint_limits.push(-999.0);
                                upper_joint_limits.push(999.0);
                            } else {
                                joint_types.push("revolute".to_string());
                                lower_joint_limits.push(joint.limits.unwrap().min);
                                upper_joint_limits.push(joint.limits.unwrap().max);
                            }
                        },
                        k::JointType::Linear { axis } => {
                            chain_joint_names.push(joint.name.clone());
                            if axis[0] == 1.0 {
                                axis_types.push("x".to_string());
                            } else if axis[1] == 1.0 {
                                axis_types.push("y".to_string());
                            } else if axis[2] == 1.0 {
                                axis_types.push("z".to_string());
                            } else if axis[0] == -1.0 {
                                axis_types.push("-x".to_string());
                            } else if axis[1] == -1.0 {
                                axis_types.push("-y".to_string());
                            } else if axis[2] == -1.0 {
                                axis_types.push("-z".to_string());
                            }
                            joint_types.push("prismatic".to_string());
                            lower_joint_limits.push(joint.limits.unwrap().min);
                            upper_joint_limits.push(joint.limits.unwrap().max);
                        }
                    }
                }

                displacements.push(joint.origin().translation.vector);
                rot_offsets.push(joint.origin().rotation);
            });
            let arm: arm::Arm = arm::Arm::init(axis_types.clone(), displacements.clone(),
            rot_offsets.clone(), joint_types.clone());
            arms.push(arm);
            chain_lengths.push(axis_types.len() as usize);
            num_dofs += axis_types.len();
            joint_names_per_chain.push(chain_joint_names);
        }

        let mut all_joint_names: Vec<String> = Vec::new();
        for chain_names in &joint_names_per_chain {
            all_joint_names.extend(chain_names.iter().cloned());
        }

        // Detect shared joints: joints with the same URDF name across different chains
        let mut joint_global_indices: HashMap<String, Vec<usize>> = HashMap::new();
        let mut global_offset = 0;
        for i in 0..num_chains {
            for j in 0..joint_names_per_chain[i].len() {
                joint_global_indices
                    .entry(joint_names_per_chain[i][j].clone())
                    .or_insert_with(Vec::new)
                    .push(global_offset + j);
            }
            global_offset += chain_lengths[i];
        }

        let mut shared_joint_pairs: Vec<(usize, usize)> = Vec::new();
        for (_name, indices) in &joint_global_indices {
            if indices.len() > 1 {
                for k in 0..indices.len() - 1 {
                    for l in k + 1..indices.len() {
                        shared_joint_pairs.push((indices[k], indices[l]));
                    }
                }
            }
        }
        shared_joint_pairs.sort();

        // Build full-to-reduced DOF mapping (shared joints map to the same unique index)
        let mut full_to_reduced = vec![0usize; num_dofs];
        let mut name_to_unique: HashMap<String, usize> = HashMap::new();
        let mut unique_counter = 0;
        global_offset = 0;
        for i in 0..num_chains {
            for j in 0..joint_names_per_chain[i].len() {
                let gidx = global_offset + j;
                if let Some(&uid) = name_to_unique.get(&joint_names_per_chain[i][j]) {
                    full_to_reduced[gidx] = uid;
                } else {
                    name_to_unique.insert(joint_names_per_chain[i][j].clone(), unique_counter);
                    full_to_reduced[gidx] = unique_counter;
                    unique_counter += 1;
                }
            }
            global_offset += chain_lengths[i];
        }
        let num_unique_dofs = unique_counter;

        // Reduced joint limits: intersection (tightest) of limits for shared joints
        let mut reduced_lower = vec![f64::NEG_INFINITY; num_unique_dofs];
        let mut reduced_upper = vec![f64::INFINITY; num_unique_dofs];
        for i in 0..num_dofs {
            let r = full_to_reduced[i];
            if reduced_lower[r] == f64::NEG_INFINITY {
                reduced_lower[r] = lower_joint_limits[i];
            } else {
                reduced_lower[r] = reduced_lower[r].max(lower_joint_limits[i]);
            }
            if reduced_upper[r] == f64::INFINITY {
                reduced_upper[r] = upper_joint_limits[i];
            } else {
                reduced_upper[r] = reduced_upper[r].min(upper_joint_limits[i]);
            }
        }

        if !shared_joint_pairs.is_empty() {
            println!("Detected {} shared joint pair(s):", shared_joint_pairs.len());
            for &(a, b) in &shared_joint_pairs {
                println!("  x[{}] ({}) <-> x[{}] ({})", a, all_joint_names[a], b, all_joint_names[b]);
            }
            println!("  Total DOFs: {}, Unique DOFs: {}", num_dofs, num_unique_dofs);
        }

        Robot{arms, num_chains, chain_lengths, num_dofs, lower_joint_limits, upper_joint_limits,
            all_joint_names, joint_names_per_chain, shared_joint_pairs, full_to_reduced,
            num_unique_dofs, reduced_lower_limits: reduced_lower, reduced_upper_limits: reduced_upper}

    }

    pub fn has_shared_joints(&self) -> bool {
        !self.shared_joint_pairs.is_empty()
    }

    /// Expand reduced-space variables to full-space (duplicate shared joint values).
    pub fn expand_to_full(&self, x_reduced: &[f64]) -> Vec<f64> {
        let mut x_full = vec![0.0; self.num_dofs];
        for i in 0..self.num_dofs {
            x_full[i] = x_reduced[self.full_to_reduced[i]];
        }
        x_full
    }

    /// Average full-space variables down to reduced-space (shared joints are averaged).
    pub fn reduce_to_unique(&self, x_full: &[f64]) -> Vec<f64> {
        let mut x_reduced = vec![0.0; self.num_unique_dofs];
        let mut counts = vec![0usize; self.num_unique_dofs];
        for i in 0..self.num_dofs {
            let r = self.full_to_reduced[i];
            x_reduced[r] += x_full[i];
            counts[r] += 1;
        }
        for i in 0..self.num_unique_dofs {
            x_reduced[i] /= counts[i] as f64;
        }
        x_reduced
    }

    /// Aggregate full-space gradient to reduced-space (sum contributions from shared joints).
    pub fn reduce_gradient(&self, grad_full: &[f64]) -> Vec<f64> {
        let mut grad_reduced = vec![0.0; self.num_unique_dofs];
        for i in 0..self.num_dofs {
            grad_reduced[self.full_to_reduced[i]] += grad_full[i];
        }
        grad_reduced
    }

    pub fn get_frames_immutable(&self, x: &[f64]) -> Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)> {
        let mut out: Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)> = Vec::new();
        let mut l = 0;
        let mut r = 0;
        for i in 0..self.num_chains {
            r += self.chain_lengths[i];
            out.push( self.arms[i].get_frames_immutable( &x[l..r] ) );
            l = r;
        }
        out
    }
    
    pub fn get_manipulability_immutable(&self, x: &[f64]) -> f64 {
        let mut out = 0.0;
        let mut l = 0;
        let mut r = 0;
        for i in 0..self.num_chains {
            r += self.chain_lengths[i];
            out += self.arms[i].get_manipulability_immutable( &x[l..r] );
            l = r;
        }
        out
    }

    pub fn get_ee_pos_and_quat_immutable(&self, x: &[f64]) -> Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)> {
        let mut out: Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)> = Vec::new();
        let mut l = 0;
        let mut r = 0;
        for i in 0..self.num_chains {
            r += self.chain_lengths[i];
            out.push( self.arms[i].get_ee_pos_and_quat_immutable( &x[l..r] ));
            l = r;
        }
        out
    }
}

