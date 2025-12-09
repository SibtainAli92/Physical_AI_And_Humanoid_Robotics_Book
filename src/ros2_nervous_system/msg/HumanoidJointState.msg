# HumanoidJointState.msg
# Custom message for humanoid robot joint states

std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
float64[] commanded_position
float64[] commanded_velocity
float64[] commanded_effort
float64[] max_effort
bool[] is_homed
bool[] is_calibrated