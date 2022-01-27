# Run Gazebo RL
1. roscore
2. roslaunch gazebo_ros empty_world.launch
3. roslaunch scan_img lscm.launch
4. python dqn_run.py

# Run Isaac RL
1. roscore
2. roslaunch scan_img lscm.launch
3. source /path/to/isaac_sim-2021.2.0/setup_python_env.sh
4. /path/to/isaac_sim-2021.2.0/python.sh dqn_run_isaac.py
