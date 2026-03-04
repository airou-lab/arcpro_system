from isaacsim import SimulationApp
import os

# Initialize simulation app
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdPhysics, Sdf

# These are the actual files causing the warnings
target_files = [
    "/home/arika/Documents/arcpro/arcpro_system/src/examples/ARCPro_RL/arc_rl_isacc_sim/f1tenth_trainer/assets/Make_Asset.usd",
    "/home/arika/Documents/arcpro/arcpro_system/src/examples/ARCPro_RL/arc_rl_isacc_sim/isacc_sim_usd/SubUSDs/jetracer_track.usd",
    "/home/arika/Documents/arcpro/arcpro_system/src/examples/ARCPro_RL/arc_rl_isacc_sim/isacc_sim_usd/World0.usd"
]

def fix_usd_file(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping (not found): {file_path}")
        return

    print(f"\n--- Cleaning: {file_path} ---")
    stage = Usd.Stage.Open(file_path)
    
    if not stage:
        print(f"Error: Could not open {file_path}")
        return

    # 1. Clear broken Material Relationships and References
    for prim in stage.Traverse():
        # Clear material bindings (the Rubber_Asphalt error)
        if prim.HasRelationship("material:binding:physics"):
            rel = prim.GetRelationship("material:binding:physics")
            print(f"  - Clearing material binding on: {prim.GetPath()}")
            rel.ClearTargets(True)

        # Clear broken References (the default_environment.usd error)
        if prim.HasAuthoredReferences():
            if "defaultGroundPlane" in str(prim.GetPath()):
                print(f"  - Removing broken reference on: {prim.GetPath()}")
                prim.GetReferences().ClearReferences()

    # 2. Remove Legacy Physics Attributes (the Stiffness/Damping warnings)
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            for attr_name in ["physics:stiffness", "physics:damping"]:
                attr = prim.GetAttribute(attr_name)
                if attr.IsValid():
                    print(f"  - Removing legacy {attr_name} from: {prim.GetPath()}")
                    prim.RemoveProperty(attr_name)

    # 3. Save
    stage.GetRootLayer().Save()
    print(f"Done cleaning {file_path}")

if __name__ == "__main__":
    for f in target_files:
        fix_usd_file(f)
    
    print("\nAll target assets processed.")
    simulation_app.close()
