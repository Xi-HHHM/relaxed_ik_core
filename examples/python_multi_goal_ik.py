#!/usr/bin/env python3
"""
Example: solve IK for multiple Cartesian goals using the Python wrapper.

Demonstrates:
  - Loading a robot from a YAML settings file (which references a URDF)
  - Solving position IK for a sequence of Cartesian goals
  - Handling multi-chain robots (e.g. Baxter with two arms)
  - Using the shared-joint API when chains overlap

Usage (from the project root, after `cargo build`):
  python examples/python_multi_goal_ik.py                           # default: configs/settings.yaml
  python examples/python_multi_goal_ik.py configs/example_settings/ur5.yaml
  python examples/python_multi_goal_ik.py configs/example_settings/baxter.yaml
  
  # Get the optimization report mode from the command line
  python examples/python_multi_goal_ik.py --report_mode="brief"
"""

import sys
import os
import math
import yaml
from argparse import ArgumentParser

# ── locate & import the wrapper ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, os.path.join(project_root, "wrappers"))

from python_wrapper import RelaxedIKRust


def load_settings(yaml_path):
    """Read the YAML settings file and return a dict with parsed fields."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return {
        "urdf":            cfg.get("urdf", ""),
        "base_links":      cfg.get("base_links", []),
        "ee_links":        cfg.get("ee_links", []),
        "starting_config": cfg.get("starting_config", []),
        "num_chains":      len(cfg.get("base_links", [])),
    }


def print_settings(settings, yaml_path):
    print("=" * 60)
    print(f"Settings file : {yaml_path}")
    print(f"URDF          : {settings['urdf']}")
    print(f"Chains        : {settings['num_chains']}")
    for i in range(settings["num_chains"]):
        print(f"  Chain {i}: {settings['base_links'][i]} -> {settings['ee_links'][i]}")
    print(f"Starting config ({len(settings['starting_config'])} DOFs): "
          f"{[round(v, 4) for v in settings['starting_config']]}")
    print("=" * 60)


def identity_quat():
    """Return identity quaternion in xyzw order."""
    return [0.0, 0.0, 0.0, 1.0]


def solve_single_chain_demo(rik, settings):
    """
    Single-chain demo: solve IK for a sequence of goals that trace a small
    circle in the YZ plane relative to the starting EE pose.
    """
    print("\n--- Single-chain IK: tracing a circle in YZ plane ---")

    num_chains = settings["num_chains"]
    start_cfg = settings["starting_config"]

    # Reset to starting configuration
    rik.reset(start_cfg)

    # Use the starting EE pose as the centre of the circle.
    # We don't have FK on the Python side, so we send solve_position once with
    # the starting config's EE pose to "warm up", then perturb from there.
    # For this demo we use hand-picked base positions per robot.
    # In practice you'd obtain the initial EE pose from your own FK or TF tree.

    # -- generate a circle of Cartesian goals for chain 0 --
    radius = 0.05  # 5 cm circle
    num_waypoints = 12
    goals = []
    for k in range(num_waypoints):
        angle = 2.0 * math.pi * k / num_waypoints
        dy = radius * math.cos(angle)
        dz = radius * math.sin(angle)
        goals.append((0.0, dy, dz))

    print(f"  Solving {num_waypoints} waypoints (radius={radius} m) ...")

    for k, (dx, dy, dz) in enumerate(goals):
        # Use solve_velocity: incremental delta from previous pose
        # Deltas are relative to the current goal, so we compute the step
        if k == 0:
            prev_dy, prev_dz = 0.0, 0.0
        else:
            prev_angle = 2.0 * math.pi * (k - 1) / num_waypoints
            prev_dy = radius * math.cos(prev_angle)
            prev_dz = radius * math.sin(prev_angle)

        step_dy = dy - prev_dy
        step_dz = dz - prev_dz

        # Build flat arrays for all chains (even if we only move chain 0)
        lin_vel = [0.0] * (3 * num_chains)
        ang_vel = [0.0] * (3 * num_chains)
        tol = [0.0] * (6 * num_chains)

        lin_vel[0] = dx          # chain 0 x
        lin_vel[1] = step_dy     # chain 0 y
        lin_vel[2] = step_dz     # chain 0 z

        solution = rik.solve_velocity(lin_vel, ang_vel, tol)
        joints = [round(solution[i], 5) for i in range(len(solution))]
        print(f"  wp {k:2d}  Δy={step_dy:+.4f} Δz={step_dz:+.4f}  joints={joints}")


def solve_multi_chain_demo(rik, settings):
    """
    Multi-chain demo (e.g. Baxter): send independent Cartesian goals to each
    chain simultaneously.
    """
    num_chains = settings["num_chains"]
    if num_chains < 2:
        print("\n--- Skipping multi-chain demo (only 1 chain) ---")
        return

    print(f"\n--- Multi-chain IK: {num_chains} chains, independent goals ---")

    rik.reset(settings["starting_config"])

    # Move each chain's EE by small increments along different axes
    steps = 8
    for step in range(steps):
        lin_vel = [0.0] * (3 * num_chains)
        ang_vel = [0.0] * (3 * num_chains)
        tol = [0.0] * (6 * num_chains)

        for c in range(num_chains):
            sign = 1.0 if c % 2 == 0 else -1.0
            lin_vel[3 * c + 1] = sign * 0.01   # y: chains move in opposite directions

        solution = rik.solve_velocity(lin_vel, ang_vel, tol)
        joints = [round(solution[i], 4) for i in range(len(solution))]
        print(f"  step {step:2d}  joints={joints}")

    # --- shared joint validation ---
    if rik.has_shared_joints():
        diffs = rik.validate_shared_joints()
        pairs = rik.get_shared_joint_pairs()
        print(f"\n  Shared joint validation ({len(pairs)} pair(s)):")
        for (a, b), d in zip(pairs, diffs):
            tag = "OK" if d < 1e-6 else "MISMATCH"
            print(f"    [{tag}] x[{a}] <-> x[{b}] : diff = {d:.2e}")


def solve_position_goals_demo(rik, settings):
    """
    Absolute-position demo: give explicit (position, orientation) goals and
    solve with solve_position.  Useful when you have target frames from a
    planner or teleoperation.
    """
    num_chains = settings["num_chains"]
    print(f"\n--- Absolute position IK for {num_chains} chain(s) ---")

    rik.reset(settings["starting_config"])

    # We'll do a few solves, each time shifting the position goal slightly.
    # Starting goal = initial EE pose (which we approximate by solving once
    # with zero tolerance and a very small perturbation).

    # First, warm up: solve with a tiny velocity to establish internal state.
    lin_vel = [0.0] * (3 * num_chains)
    ang_vel = [0.0] * (3 * num_chains)
    tol = [0.0] * (6 * num_chains)
    rik.solve_velocity(lin_vel, ang_vel, tol)

    # Now build a sequence of absolute Cartesian goals.
    # We'll shift the goal 1 cm in +x per step for each chain.
    base_positions = []
    base_quats = []
    for c in range(num_chains):
        # Approximate "current" EE pose — in a real pipeline you'd get this
        # from FK or a TF lookup.  Here we use a nominal pose per robot.
        base_positions.append([0.5, 0.0, 0.5])       # rough starting guess
        base_quats.append(identity_quat())

    # Each item: [x, y, z, qx, qy, qz, qw]
    # Order must match ee_links in your YAML.
    hardcoded_goals = [
        [0.19308424646874356, 0.3887290792692531, 0.10196842474045772, -0.4268693882499392, 0.20450339781386667, 0.23183331942302063, 0.8498318643490662],
        [0.19263177553867508, -0.3810092001730527, 0.10039854295971179, 0.4324687205104001, 0.20665035916292784, -0.22553012351571247, 0.848176042001257],
    ]

    positions_flat = []
    quats_flat = []
    tol_flat = []

    for c in range(len(hardcoded_goals)):
        positions_flat.extend(hardcoded_goals[c][:3])
        quats_flat.extend(hardcoded_goals[c][3:])
        tol_flat.extend([0.0] * 6)

    solution = rik.solve_position(positions_flat, quats_flat, tol_flat)
    joints = [round(solution[i], 4) for i in range(len(solution))]
    print(f"  joints={joints}")

    # Validate shared joints
    if rik.has_shared_joints():
        diffs = rik.validate_shared_joints()
        pairs = rik.get_shared_joint_pairs()
        print(f"\n  Shared joint validation ({len(pairs)} pair(s)):")
        for (a, b), d in zip(pairs, diffs):
            val_a = round(solution[a], 6)
            val_b = round(solution[b], 6)
            tag = "OK" if d < 1e-4 else "MISMATCH"
            print(f"    [{tag}] x[{a}]={val_a} vs x[{b}]={val_b}, diff={d:.2e}")


def main():
    # --- pick settings file ---
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML file, or config name (e.g. agileone_upper for configs/example_settings/agileone_upper.yaml)"
    )
    arg_parser.add_argument(
        "--report_mode",
        type=str,
        choices=["off", "brief", "detailed"],
        default="off",
        help="Objective report verbosity after each solve"
    )
    args = arg_parser.parse_args()

    if args.config is None:
        yaml_path = os.path.join(project_root, "configs", "settings.yaml")
    elif "/" in args.config or args.config.endswith(".yaml"):
        yaml_path = args.config
    else:
        yaml_path = os.path.join(project_root, "configs", "example_settings", args.config + ".yaml")

    optimization_report_mode = args.report_mode

    if not os.path.isabs(yaml_path):
        yaml_path = os.path.abspath(yaml_path)

    settings = load_settings(yaml_path)
    print_settings(settings, yaml_path)

    # --- create solver ---
    rik = RelaxedIKRust(yaml_path)
    rik.set_objective_report_mode(optimization_report_mode)

    # --- shared joint setup (call BEFORE solving) ---
    if rik.has_shared_joints():
        pairs = rik.get_shared_joint_pairs()
        print(f"\nShared joints detected: {len(pairs)} pair(s)")
        for a, b in pairs:
            print(f"  x[{a}] <-> x[{b}]")

        # Uncomment ONE of the two approaches:
        # rik.enable_shared_joint_penalty(weight=2000000.0)  # Approach 2.1: weight is the direct penalty coefficient
        rik.enable_shared_joint_reduction()                   # Approach 2.2: exact, no tuning needed

    # --- demos ---
    # solve_single_chain_demo(rik, settings)
    # solve_multi_chain_demo(rik, settings)
    solve_position_goals_demo(rik, settings)

    print("\nDone.")


if __name__ == "__main__":
    main()
