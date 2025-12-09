// safety_monitor.cpp
// C++ implementation of safety monitoring system for humanoid robot

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <memory>

using std::placeholders::_1;

class SafetyMonitor : public rclcpp::Node
{
public:
    SafetyMonitor() : Node("safety_monitor_cpp")
    {
        // Declare parameters
        this->declare_parameter("emergency_stop_timeout", 0.1);
        this->declare_parameter("max_joint_velocity", 5.0);
        this->declare_parameter("max_joint_effort", 100.0);
        this->declare_parameter("max_angular_velocity", 3.0);
        this->declare_parameter("max_linear_acceleration", 20.0);
        this->declare_parameter("collision_force_threshold", 50.0);
        this->declare_parameter("fall_angle_threshold", 0.5);

        // Get parameters
        this->get_parameter("emergency_stop_timeout", emergency_stop_timeout_);
        this->get_parameter("max_joint_velocity", max_joint_velocity_);
        this->get_parameter("max_joint_effort", max_joint_effort_);
        this->get_parameter("max_angular_velocity", max_angular_velocity_);
        this->get_parameter("max_linear_acceleration", max_linear_acceleration_);
        this->get_parameter("collision_force_threshold", collision_force_threshold_);
        this->get_parameter("fall_angle_threshold", fall_angle_threshold_);

        // Create subscribers
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&SafetyMonitor::joint_state_callback, this, _1));

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data", 10,
            std::bind(&SafetyMonitor::imu_callback, this, _1));

        ft_sub_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
            "/left_foot/ft_sensor", 10,
            std::bind(&SafetyMonitor::ft_callback, this, _1));

        right_ft_sub_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
            "/right_foot/ft_sensor", 10,
            std::bind(&SafetyMonitor::ft_callback, this, _1));

        // Create publisher
        emergency_stop_pub_ = this->create_publisher<std_msgs::msg::Bool>("/emergency_stop", 10);

        // Initialize state variables
        last_joint_state_time_ = this->get_clock()->now();
        emergency_stop_active_ = false;
        fall_detected_ = false;
        collision_detected_ = false;

        // Create timer for safety checks
        safety_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),  // 100Hz
            std::bind(&SafetyMonitor::safety_check, this));

        RCLCPP_INFO(this->get_logger(), "Safety Monitor (C++) initialized");
    }

private:
    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        last_joint_state_time_ = this->get_clock()->now();

        for (size_t i = 0; i < msg->name.size(); ++i) {
            if (i < msg->position.size()) {
                current_joint_positions_[msg->name[i]] = msg->position[i];
            }
            if (i < msg->velocity.size()) {
                current_joint_velocities_[msg->name[i]] = msg->velocity[i];
            }
            if (i < msg->effort.size()) {
                current_joint_efforts_[msg->name[i]] = msg->effort[i];
            }
        }
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        // Check angular velocity limits
        double angular_vel = std::sqrt(
            msg->angular_velocity.x * msg->angular_velocity.x +
            msg->angular_velocity.y * msg->angular_velocity.y +
            msg->angular_velocity.z * msg->angular_velocity.z
        );

        if (angular_vel > max_angular_velocity_) {
            RCLCPP_WARN(this->get_logger(),
                "Angular velocity limit exceeded: %f > %f",
                angular_vel, max_angular_velocity_);
            trigger_emergency_stop();
        }

        // Check linear acceleration limits
        double linear_acc = std::sqrt(
            msg->linear_acceleration.x * msg->linear_acceleration.x +
            msg->linear_acceleration.y * msg->linear_acceleration.y +
            msg->linear_acceleration.z * msg->linear_acceleration.z
        );

        if (linear_acc > max_linear_acceleration_) {
            RCLCPP_WARN(this->get_logger(),
                "Linear acceleration limit exceeded: %f > %f",
                linear_acc, max_linear_acceleration_);
            trigger_emergency_stop();
        }

        // Check for potential fall (simplified: check roll/pitch angles)
        tf2::Quaternion quat(
            msg->orientation.x,
            msg->orientation.y,
            msg->orientation.z,
            msg->orientation.w
        );
        tf2::Matrix3x3 mat(quat);
        double roll, pitch, yaw;
        mat.getRPY(roll, pitch, yaw);

        if (std::abs(roll) > fall_angle_threshold_ || std::abs(pitch) > fall_angle_threshold_) {
            RCLCPP_WARN(this->get_logger(),
                "Potential fall detected: roll=%f, pitch=%f", roll, pitch);
            fall_detected_ = true;
        }
    }

    void ft_callback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
    {
        double force_magnitude = std::sqrt(
            msg->wrench.force.x * msg->wrench.force.x +
            msg->wrench.force.y * msg->wrench.force.y +
            msg->wrench.force.z * msg->wrench.force.z
        );

        if (force_magnitude > collision_force_threshold_) {
            RCLCPP_WARN(this->get_logger(),
                "Collision detected: force=%f > %f",
                force_magnitude, collision_force_threshold_);
            collision_detected_ = true;
            trigger_emergency_stop();
        }
    }

    void safety_check()
    {
        if (emergency_stop_active_) {
            return;
        }

        // Check for joint limit violations
        for (const auto& pair : current_joint_velocities_) {
            if (std::abs(pair.second) > max_joint_velocity_) {
                RCLCPP_WARN(this->get_logger(),
                    "Joint velocity limit exceeded for %s: %f > %f",
                    pair.first.c_str(), pair.second, max_joint_velocity_);
            }
        }

        for (const auto& pair : current_joint_efforts_) {
            if (std::abs(pair.second) > max_joint_effort_) {
                RCLCPP_WARN(this->get_logger(),
                    "Joint effort limit exceeded for %s: %f > %f",
                    pair.first.c_str(), pair.second, max_joint_effort_);
            }
        }

        // Check for communication timeout
        auto time_since_last_joint_state =
            (this->get_clock()->now() - last_joint_state_time_).seconds();

        if (time_since_last_joint_state > emergency_stop_timeout_) {
            RCLCPP_ERROR(this->get_logger(),
                "Joint state timeout - emergency stop triggered");
            trigger_emergency_stop();
        }

        // Check for fall detection
        if (fall_detected_) {
            RCLCPP_WARN(this->get_logger(), "Fall detected - emergency stop triggered");
            trigger_emergency_stop();
        }

        // Check for collision detection
        if (collision_detected_) {
            RCLCPP_WARN(this->get_logger(), "Collision detected - emergency stop triggered");
            trigger_emergency_stop();
        }
    }

    void trigger_emergency_stop()
    {
        if (!emergency_stop_active_) {
            emergency_stop_active_ = true;
            auto msg = std_msgs::msg::Bool();
            msg.data = true;
            emergency_stop_pub_->publish(msg);
            RCLCPP_ERROR(this->get_logger(), "EMERGENCY STOP ACTIVATED (C++)");
        }
    }

    // Member variables
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr ft_sub_;
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr right_ft_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr emergency_stop_pub_;
    rclcpp::TimerBase::SharedPtr safety_timer_;

    std::unordered_map<std::string, double> current_joint_positions_;
    std::unordered_map<std::string, double> current_joint_velocities_;
    std::unordered_map<std::string, double> current_joint_efforts_;
    rclcpp::Time last_joint_state_time_;

    double emergency_stop_timeout_;
    double max_joint_velocity_;
    double max_joint_effort_;
    double max_angular_velocity_;
    double max_linear_acceleration_;
    double collision_force_threshold_;
    double fall_angle_threshold_;

    bool emergency_stop_active_;
    bool fall_detected_;
    bool collision_detected_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SafetyMonitor>());
    rclcpp::shutdown();
    return 0;
}