#! /usr/bin/env python3

import ctypes
import os

class Opt(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)), ("length", ctypes.c_int)]

class RelaxedIKS(ctypes.Structure):
    pass

dir_path = os.path.dirname(os.path.realpath(__file__))
lib = ctypes.cdll.LoadLibrary(dir_path + '/../target/debug/librelaxed_ik_lib.so')

lib.relaxed_ik_new.restype = ctypes.POINTER(RelaxedIKS)
lib.solve.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.solve.restype = Opt
lib.solve_position.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.solve_position.restype = Opt
lib.solve_velocity.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.solve_velocity.restype = Opt
lib.reset.argtypes = [ctypes.POINTER(RelaxedIKS)]

# Shared joint API
lib.has_shared_joints.argtypes = [ctypes.POINTER(RelaxedIKS)]
lib.has_shared_joints.restype = ctypes.c_int
lib.get_num_shared_joint_pairs.argtypes = [ctypes.POINTER(RelaxedIKS)]
lib.get_num_shared_joint_pairs.restype = ctypes.c_int
lib.get_shared_joint_pairs.argtypes = [ctypes.POINTER(RelaxedIKS)]
lib.get_shared_joint_pairs.restype = Opt
lib.enable_shared_joint_penalty.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.c_double]
lib.enable_shared_joint_penalty.restype = None
lib.enable_shared_joint_reduction.argtypes = [ctypes.POINTER(RelaxedIKS)]
lib.enable_shared_joint_reduction.restype = None
lib.enable_relative_tcp_constraints.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.c_double]
lib.enable_relative_tcp_constraints.restype = None
lib.set_objective_report_mode.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.c_int]
lib.set_objective_report_mode.restype = None
lib.set_max_iterations.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.c_int]
lib.set_max_iterations.restype = None
lib.get_max_iterations.argtypes = [ctypes.POINTER(RelaxedIKS)]
lib.get_max_iterations.restype = ctypes.c_int
lib.set_objective_weight.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.c_char_p, ctypes.c_double]
lib.set_objective_weight.restype = None
lib.validate_shared_joints.argtypes = [ctypes.POINTER(RelaxedIKS)]
lib.validate_shared_joints.restype = Opt

class RelaxedIKRust:
    def __init__(self, setting_file_path = None):
        '''
        setting_file_path (string): path to the setting file
                                    if no path is given, the default setting file will be used
                                    /configs/settings.yaml
        '''
        if setting_file_path is None:
            self.obj = lib.relaxed_ik_new(ctypes.c_char_p())
        else:
            self.obj = lib.relaxed_ik_new(ctypes.c_char_p(setting_file_path.encode('utf-8')))
    
    def __exit__(self, exc_type, exc_value, traceback):
        lib.relaxed_ik_free(self.obj)
    
    def solve_position(self, positions, orientations, tolerances):
        '''
        Assuming the robot has N end-effectors
        positions (1D array with length as 3*N): list of end-effector positions
        orientations (1D array with length as 4*N): list of end-effector orientations (in quaternion xyzw format)
        tolerances (1D array with length as 6*N): list of tolerances for each end-effector (x, y, z, rx, ry, rz)
        '''
        pos_arr = (ctypes.c_double * len(positions))()
        quat_arr = (ctypes.c_double * len(orientations))()
        tole_arr = (ctypes.c_double * len(tolerances))()
        for i in range(len(positions)):
            pos_arr[i] = positions[i]
        for i in range(len(orientations)):
            quat_arr[i] = orientations[i]
        for i in range(len(tolerances)):
            tole_arr[i] = tolerances[i]
        xopt = lib.solve_position(self.obj, pos_arr, len(pos_arr), quat_arr, len(quat_arr), tole_arr, len(tole_arr))
        return xopt.data[:xopt.length]
    
    def solve_velocity(self, linear_velocities, angular_velocities, tolerances):
        '''
        Assuming the robot has N end-effectors
        linear_velocities (1D array with length as 3*N): list of end-effector linear velocities
        angular_velocities (1D array with length as 4*N): list of end-effector angular velocities
        tolerances (1D array with length as 6*N): list of tolerances for each end-effector (x, y, z, rx, ry, rz)
        '''
        linear_arr = (ctypes.c_double * len(linear_velocities))()
        angular_arr = (ctypes.c_double * len(angular_velocities))()
        tole_arr = (ctypes.c_double * len(tolerances))()
        for i in range(len(linear_velocities)):
            linear_arr[i] = linear_velocities[i]
        for i in range(len(angular_velocities)):
            angular_arr[i] = angular_velocities[i]
        for i in range(len(tolerances)):
            tole_arr[i] = tolerances[i]
        xopt = lib.solve_velocity(self.obj, linear_arr, len(linear_arr), angular_arr, len(angular_arr), tole_arr, len(tole_arr))
        return xopt.data[:xopt.length]
    
    def reset(self, joint_state):
        js_arr = (ctypes.c_double * len(joint_state))()
        for i in range(len(joint_state)):
            js_arr[i] = joint_state[i]
        lib.reset(self.obj, js_arr, len(js_arr))

    # ---- Shared joint API ----

    def has_shared_joints(self):
        '''Returns True if the robot has shared joints between kinematic chains.'''
        return lib.has_shared_joints(self.obj) != 0

    def get_shared_joint_pairs(self):
        '''
        Returns a list of (idx_a, idx_b) tuples — pairs of indices into the
        full joint-state vector that refer to the same physical joint.
        '''
        opt = lib.get_shared_joint_pairs(self.obj)
        flat = [opt.data[i] for i in range(opt.length)]
        return [(int(flat[i]), int(flat[i+1])) for i in range(0, len(flat), 2)]

    def enable_shared_joint_penalty(self, weight=2000000.0):
        '''
        Approach 2.1: Add penalty terms that force shared joint variables to
        stay aligned.  Cost per pair = weight * (x[a] - x[b])^2.

        weight (float): penalty coefficient (default 2e6).
                        Must be large enough to dominate position objectives
                        (which total ~600 effective weight across chains).
                        Typical range: 1e4 – 1e7.
        '''
        lib.enable_shared_joint_penalty(self.obj, weight)

    def enable_shared_joint_reduction(self):
        '''
        Approach 2.2: Optimize in a reduced variable space where each physical
        joint has exactly one optimization variable.  Shared joints are
        guaranteed to be identical after solving.
        '''
        lib.enable_shared_joint_reduction(self.obj)

    def enable_relative_tcp_constraints(self, weight=100.0):
        '''
        Keep end-effectors in the relative pose implied by current goals.
        Useful when absolute goal tracking can be relaxed but the formation
        (relative geometry between TCPs) must be maintained. Requires 2+ chains.
        weight (float): objective weight (default 100). Tune vs MatchEEPosiDoF/MatchEERotaDoF.
        '''
        lib.enable_relative_tcp_constraints(self.obj, weight)

    def set_objective_report_mode(self, mode='off'):
        '''
        Set objective report verbosity after each solve.
        mode: 'off' (default), 'brief', or 'detailed'
          - off: no reports
          - brief: one row per objective class
          - detailed: one row per objective instance
        '''
        m = {'off': 0, 'brief': 1, 'detailed': 2}.get(mode.lower(), 0)
        lib.set_objective_report_mode(self.obj, m)

    def set_max_iterations(self, max_iterations=100):
        '''
        Set optimizer iteration budget per solve call.
        max_iterations (int): must be >= 1
        '''
        max_iterations = int(max_iterations)
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        lib.set_max_iterations(self.obj, max_iterations)

    def get_max_iterations(self):
        '''Return current optimizer iteration budget per solve call.'''
        return int(lib.get_max_iterations(self.obj))

    def set_objective_weight(self, class_name, weight):
        '''
        Set weight for all objectives of a given class.
        class_name: one of MatchEEPosiDoF, MatchEERotaDoF, EachJointLimits,
                    MinimizeVelocity, MinimizeAcceleration, MinimizeJerk,
                    MaximizeManipulability, SelfCollision, RelativeTCPConstraint
        weight (float): the weight to use
        '''
        lib.set_objective_weight(self.obj, class_name.encode('utf-8'), weight)

    def validate_shared_joints(self):
        '''
        After solving, returns a list of absolute differences for each shared
        joint pair (based on the current internal solution).
        For Penalty mode, small nonzero values are expected.
        For VariableReduction mode, values should be exactly 0.
        '''
        opt = lib.validate_shared_joints(self.obj)
        return [opt.data[i] for i in range(opt.length)]

if __name__ == '__main__':
    pass
