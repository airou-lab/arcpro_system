https://airou-lab.github.io/arcpro_ros2_website/

To clone run:
```bash
git clone --recurse-submodules -j8  git@github.com:airou-lab/arcpro_system.git
```

(To add build passing tags)

(To add current version deployment)

(To add current maintainer)

[//]: # (git submodule add -f git@github.com:airou-lab/twist_to_ackermann.git src/base/twist_to_ackermann)

Maintainer until Dec 2027: arikak@ou.edu

When building on new device run in project root:
```bash
rosdep install --from-paths src -y --ignore-src
```


# Running Examples:
```bash
# waypointer example:
 ./waypoint.sh 
 # RL example:
./arcpro_rl.sh 
```

to just view the xacro file, run 
```bash
/home/arika/Documents/arcpro/arcpro_system/src/base/robot urdf models/rsp_xacro_test.sh

# AND in host terminal for gui
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```