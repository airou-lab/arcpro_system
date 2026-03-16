import re
import os

def verify_config():
    cfg_path = "src/examples/ARCPro_RL/arc_rl_isacc_sim/arcproLab/arcpro_env_cfg.py"
    if not os.path.exists(cfg_path):
        print(f"FAIL: {cfg_path} not found")
        return False
    
    with open(cfg_path, "r") as f:
        content = f.read()
    
    # Check track position
    track_pos_match = re.search(r"track\s*=\s*AssetBaseCfg\(.*?init_state=AssetBaseCfg\.InitialStateCfg\(pos=\((.*?)\)\)", content, re.DOTALL)
    if track_pos_match:
        pos = track_pos_match.group(1).strip()
        if pos == "0.0, 0.0, 0.0":
            print(f"PASS: Track position is {pos}")
        else:
            print(f"FAIL: Track position is {pos}, expected 0.0, 0.0, 0.0")
    else:
        print("FAIL: Could not find track position in config")
        return False

    # Check robot position
    robot_pos_match = re.search(r"robot\s*=\s*ARCPRO_ROBOT_CFG\.replace\(.*?init_state=ARCPRO_ROBOT_CFG\.init_state\.replace\(pos=\((.*?)\)\)", content, re.DOTALL)
    if robot_pos_match:
        pos = robot_pos_match.group(1).strip()
        if pos == "0.0, 0.0, 0.05":
            print(f"PASS: Robot position is {pos}")
        else:
            print(f"FAIL: Robot position is {pos}, expected 0.0, 0.0, 0.05")
    else:
        print("FAIL: Could not find robot position in config")
        return False

    return True

if __name__ == "__main__":
    if verify_config():
        print("Verification SUCCESS: Configured positions are correct.")
        exit(0)
    else:
        print("Verification FAILURE: Configured positions do not match requirements.")
        exit(1)
