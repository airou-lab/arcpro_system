from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdPhysics, PhysxSchema

usd_path = "/home/arika/Documents/arcpro/arcpro_system/src/examples/ARCPro_RL/arc_rl_isacc_sim/isacc_sim_usd/World0.usd"
omni.usd.get_context().open_stage(usd_path)
stage = omni.usd.get_context().get_stage()

def check_prim_physics(path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(f"[FAIL] {path} is NOT valid")
        return
    
    has_physics = prim.HasAPI(UsdPhysics.CollisionAPI)
    has_physx = prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
    print(f"Path: {path}")
    print(f"  - Has UsdPhysics.CollisionAPI: {has_physics}")
    print(f"  - Has PhysxSchema.PhysxCollisionAPI: {has_physx}")
    
    if has_physx:
        physx_api = PhysxSchema.PhysxCollisionAPI(prim)
        contact_offset = physx_api.GetContactOffsetAttr().Get()
        rest_offset = physx_api.GetRestOffsetAttr().Get()
        print(f"  - Contact Offset: {contact_offset}")
        print(f"  - Rest Offset: {rest_offset}")

print("\n--- Diagnostic: Collision & Physics Check ---")

# 1. Check the Track/Floor
print("\n[Checking Track/Floor]")
track_path = "/World/jetracer_track"
check_prim_physics(track_path)
track_prim = stage.GetPrimAtPath(track_path)
if track_prim.IsValid():
    for child in track_prim.GetChildren():
        if "mesh" in child.GetName().lower() or "ground" in child.GetName().lower() or "visual" in child.GetName().lower():
            check_prim_physics(child.GetPath())

# 2. Check the Wheels
print("\n[Checking Robot Wheels]")
wheel_paths = [
    "/World/F1Tenth/Rigid_Bodies/Wheel_Front_Left",
    "/World/F1Tenth/Rigid_Bodies/Wheel_Front_Right",
    "/World/F1Tenth/Rigid_Bodies/Wheel_Rear_Left",
    "/World/F1Tenth/Rigid_Bodies/Wheel_Rear_Right"
]
for wp in wheel_paths:
    check_prim_physics(wp)

# 3. Check Physics Scene
scene_prim = stage.GetPrimAtPath("/PhysicsScene")
if not scene_prim.IsValid():
    # Try alternate path
    scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")

if not scene_prim.IsValid():
    print("\n[WARNING] PhysicsScene NOT found at /PhysicsScene or /World/PhysicsScene!")
else:
    print(f"\n[PhysicsScene Found] at {scene_prim.GetPath()}")

simulation_app.close()
