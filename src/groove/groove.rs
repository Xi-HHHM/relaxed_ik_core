use crate::groove::gradient::{ForwardFiniteDiff, CentralFiniteDiff, GradientFinder, ForwardFiniteDiffImmutable, CentralFiniteDiffImmutable, GradientFinderImmutable};
use crate::groove::vars::{RelaxedIKVars};
use optimization_engine::{constraints::*, panoc::*, *};
use crate::groove::objective_master::ObjectiveMaster;

pub struct OptimizationEngineOpen {
    dim: usize,
    cache: PANOCCache
}
impl OptimizationEngineOpen {
    pub fn new(dim: usize) -> Self {
        let mut cache = PANOCCache::new(dim, 1e-14, 10);
        OptimizationEngineOpen { dim, cache }
    }

    pub fn optimize(&mut self, x: &mut [f64], v: &RelaxedIKVars, om: &ObjectiveMaster, max_iter: usize) {
        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let (my_obj, my_grad) = om.gradient(u, v);
            for i in 0..my_grad.len() {
                grad[i] = my_grad[i];
            }
            Ok(())
        };

        let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
            *c = om.call(u, v);
            Ok(())
        };

        // let bounds = NoConstraints::new();
        let bounds = Rectangle::new(Option::from(v.robot.lower_joint_limits.as_slice()), Option::from(v.robot.upper_joint_limits.as_slice()));

        /* PROBLEM STATEMENT */
        let problem = Problem::new(&bounds, df, f);
        let mut panoc = PANOCOptimizer::new(problem, &mut self.cache).with_max_iter(max_iter).with_tolerance(0.0005);
        // let mut panoc = PANOCOptimizer::new(problem, &mut self.cache);

        // Invoke the solver
        let status = panoc.solve(x);

        // println!("Panoc status: {:#?}", status);
        // println!("Panoc solution: {:#?}", x);
    }

    /// Optimize in reduced variable space (approach 2.2 for shared joints).
    /// `x` is in reduced space; internally expanded to full space for objective/gradient evaluation,
    /// then gradients are aggregated back to reduced space.
    pub fn optimize_reduced(&mut self, x: &mut [f64], v: &RelaxedIKVars, om: &ObjectiveMaster, max_iter: usize) {
        let robot = &v.robot;

        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let u_full = robot.expand_to_full(u);
            let (_obj, grad_full) = om.gradient(&u_full, v);
            let grad_reduced = robot.reduce_gradient(&grad_full);
            for i in 0..grad_reduced.len() {
                grad[i] = grad_reduced[i];
            }
            Ok(())
        };

        let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
            let u_full = robot.expand_to_full(u);
            *c = om.call(&u_full, v);
            Ok(())
        };

        let bounds = Rectangle::new(
            Option::from(robot.reduced_lower_limits.as_slice()),
            Option::from(robot.reduced_upper_limits.as_slice())
        );

        let problem = Problem::new(&bounds, df, f);
        let mut panoc = PANOCOptimizer::new(problem, &mut self.cache)
            .with_max_iter(max_iter)
            .with_tolerance(0.0005);
        let status = panoc.solve(x);
    }
}
