import torch
import os
import sys

# Add path to src
sys.path.append(os.path.abspath("./src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab"))

from mdp.track_manager import get_track_manager

def test_track_manager_init():
    tm = get_track_manager()
    
    # Simulate first call to compute_errors
    num_envs = 10
    pos = torch.zeros((num_envs, 3), device=tm.device)
    yaw = torch.zeros(num_envs, device=tm.device)
    
    # Initial state should be None
    assert tm.lap_count is None
    assert tm.last_wp_idx is None
    
    tm.compute_errors(pos, yaw)
    
    # State should be initialized
    assert tm.lap_count is not None
    assert tm.last_wp_idx is not None
    assert tm.lap_count.shape[0] == num_envs
    assert tm.last_wp_idx.shape[0] == num_envs
    assert tm.lap_count.dtype == torch.int32
    assert tm.last_wp_idx.dtype == torch.long
    
    print("Test passed: lap_count and last_wp_idx initialized correctly.")

if __name__ == "__main__":
    test_track_manager_init()
