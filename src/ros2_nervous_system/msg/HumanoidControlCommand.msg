# HumanoidControlCommand.msg
# Custom message for humanoid robot control commands

std_msgs/Header header
string control_mode  # position, velocity, effort, impedance
float64[] joint_positions
float64[] joint_velocities
float64[] joint_efforts
float64[] stiffness_values
float64[] damping_values
float64[] feedforward_efforts
string[] joint_names
float64 duration  # Command execution duration in seconds
bool enable_logging