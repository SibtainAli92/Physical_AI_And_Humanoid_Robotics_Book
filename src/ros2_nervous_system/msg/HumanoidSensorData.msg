# HumanoidSensorData.msg
# Custom message for humanoid robot sensor data

std_msgs/Header header
# IMU data for multiple body parts
geometry_msgs/Vector3[] imu_linear_acceleration
geometry_msgs/Vector3[] imu_angular_velocity
geometry_msgs/Quaternion[] imu_orientation
string[] imu_frame_names

# Force/torque sensor data
geometry_msgs/Wrench[] ft_wrenches
string[] ft_frame_names

# Encoder data
float64[] encoder_positions
float64[] encoder_velocities
string[] encoder_joint_names

# Temperature sensors
float64[] joint_temperatures
string[] temperature_joint_names

# Battery and power data
float64 battery_voltage
float64 battery_current
float64 battery_level
bool battery_charging
float64[] power_consumption_per_joint