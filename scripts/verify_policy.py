#!/usr/bin/env python3

import argparse
import sys
import os
import json
import glob
import time

# 1. Dependency Checks & AppLauncher Setup
try:
    import torch
except ImportError:
    print(json.dumps({"status": "error", "reason": "torch not found"}, indent=2), file=sys.stderr)
    sys.exit(1)

try:
    from isaaclab.app import AppLauncher
except ImportError:
    # Fail loudly if Isaac Lab is missing
    error_msg = {
        "status": "error",
        "reason": "Isaac Lab not found. Please run this script through ./isaaclab.sh python scripts/verify_policy.py",
    }
    print(json.dumps(error_msg, indent=2), file=sys.stderr)
    sys.exit(1)

# Add argparse arguments for Isaac Lab
parser = argparse.ArgumentParser(description='Run physical verification of the trained policy using Isaac Lab.')
parser.add_argument('--checkpoint-dir', type=str, default='logs/rsl_rl/arcpro_retraining',
                    help='Directory containing the trained policy checkpoints.')
parser.add_argument('--laps', type=int, default=10, help='Number of laps to run')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulation app (headless is often preferred for verification)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Environment & Model Imports
import gymnasium as gym
import numpy as np

# Ensure src is in the python path for imports
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    import examples.ARCPro_RL.arc_rl_isacc_sim.arcproLab as arcpro_lab
    from examples.ARCPro_RL.arc_rl_isacc_sim.arcproLab.mdp.track_manager import get_track_manager
except ImportError as e:
    print(json.dumps({
        "status": "error",
        "reason": f"Failed to import ARCPro environment components: {e}",
    }, indent=2), file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

def find_best_checkpoint(checkpoint_dir):
    """Finds the latest/best checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    pt_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not pt_files:
        return None
    # Prioritize model_*.pt
    model_files = [f for f in pt_files if "model_" in os.path.basename(f)]
    if model_files:
        try:
            model_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            return model_files[-1]
        except:
            model_files.sort()
            return model_files[-1]
    pt_files.sort(key=os.path.getmtime)
    return pt_files[-1]

def load_policy(checkpoint_path, num_obs, num_actions, device):
    """Loads rsl_rl policy from checkpoint."""
    try:
        # Check for rsl_rl dependency
        from rsl_rl.modules import ActorCritic
    except ImportError:
        print(json.dumps({"status": "error", "reason": "rsl_rl dependency not found"}, indent=2), file=sys.stderr)
        return None

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Policy architecture (hardcoded to standard RSL_RL MLP for ARCPro)
    actor_critic = ActorCritic(
        num_obs=num_obs,
        num_privileged_obs=0,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu'
    ).to(device)
    
    # rsl_rl typically saves weights under 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback to direct state dict if available
        actor_critic.load_state_dict(checkpoint)
        
    actor_critic.eval()
    return actor_critic

def main():
    # Create environment configuration
    env_cfg = arcpro_lab.arcpro_env_cfg.ARCProEnvCfg()
    env_cfg.scene.num_envs = 1  # Single environment for verification
    env_cfg.enable_hud = False  # Disable HUD for headless verification
    
    try:
        env = gym.make("ARCPro-v0", cfg=env_cfg)
    except Exception as e:
        print(json.dumps({"status": "error", "reason": f"Failed to instantiate environment: {e}"}, indent=2), file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    # Load the best available policy checkpoint
    checkpoint_path = find_best_checkpoint(args_cli.checkpoint_dir)
    if not checkpoint_path:
        print(json.dumps({"status": "error", "reason": "No policy checkpoint found in directory", "details": {"dir": args_cli.checkpoint_dir}}, indent=2), file=sys.stderr)
        env.close()
        simulation_app.close()
        sys.exit(1)

    policy = load_policy(checkpoint_path, env.num_obs, env.num_actions, args_cli.device)
    if not policy:
        env.close()
        simulation_app.close()
        sys.exit(1)

    # Physical Verification Loop
    obs, _ = env.reset()
    tm = get_track_manager(device=args_cli.device)
    tm.reset_laps(torch.tensor([0], device=args_cli.device))
    
    telemetry = []
    total_time = 0
    max_lateral_error_run = 0
    current_lap = 0
    lap_start_time = time.time()
    
    print(f"Verification started: Goal is {args_cli.laps} laps, LatErr < 0.3m")

    try:
        while current_lap < args_cli.laps:
            # Inference
            with torch.no_grad():
                actions = policy.act_inference(obs)
                
            # Physics Step
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Extract state for telemetry
            robot_pos = env.scene["robot"].data.root_pos_w
            robot_quat = env.scene["robot"].data.root_quat_w
            
            # Compute yaw for lateral error calculation
            # q = [qw, qx, qy, qz]
            yaw = torch.atan2(2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]), 
                              1.0 - 2.0 * (robot_quat[:, 2]**2 + robot_quat[:, 3]**2))
            
            lat_err, _ = tm.compute_errors(robot_pos, yaw)
            abs_lat_err = torch.abs(lat_err[0]).item()
            
            if abs_lat_err > max_lateral_error_run:
                max_lateral_error_run = abs_lat_err
                
            # Enforce lateral error bound (Requirement R002)
            if abs_lat_err > 0.3:
                failure_msg = {
                    "status": "failure",
                    "reason": "Lateral error exceeded 0.3m bound",
                    "details": {"lap": current_lap + 1, "error": abs_lat_err}
                }
                print(json.dumps(failure_msg, indent=2), file=sys.stderr)
                env.close()
                simulation_app.close()
                sys.exit(1)
                
            # Update lap counts
            laps = tm.update_laps(robot_pos)
            if laps[0] > current_lap:
                lap_end_time = time.time()
                lap_duration = lap_end_time - lap_start_time
                current_lap = laps[0].item()
                
                lap_metrics = {
                    "lap": current_lap,
                    "max_lateral_error": abs_lat_err, 
                    "time": lap_duration
                }
                telemetry.append(lap_metrics)
                total_time += lap_duration
                lap_start_time = lap_end_time
                print(f"Lap {current_lap} completed: time={lap_duration:.2f}s, lat_err={abs_lat_err:.4f}m")

            # Check for crashes/termination
            if terminated.any() or truncated.any():
                failure_msg = {
                    "status": "failure",
                    "reason": "Robot crashed or left track area",
                    "details": {"lap": current_lap + 1}
                }
                print(json.dumps(failure_msg, indent=2), file=sys.stderr)
                env.close()
                simulation_app.close()
                sys.exit(1)

    except KeyboardInterrupt:
        print("Verification interrupted by user.")
    
    # Final Result
    summary = {
        "status": "success",
        "total_laps": current_lap,
        "total_time": total_time,
        "max_lateral_error": max_lateral_error_run,
        "laps": telemetry
    }
    
    print(json.dumps(summary, indent=2))
    
    env.close()
    simulation_app.close()
    sys.exit(0)

if __name__ == '__main__':
    main()
