// joint_limit_enforcer.cpp
// System to enforce joint limits and constraints for humanoid robot

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <control_msgs/msg/joint_trajectory_controller_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <std_msgs/msg/bool.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <memory>

using std::placeholders::_1;

struct JointLimits {
    double min_position;
    double max_position;
    double max_velocity;
    double max_effort;
};

class JointLimitEnforcer : public rclcpp::Node
{
public:
    JointLimitEnforcer() : Node("joint_limit_enforcer")
    {
        // Declare parameters
        this->declare_parameter("position_safety_margin", 0.05);
        this->declare_parameter("velocity_safety_factor", 0.9);
        this->declare_parameter("effort_safety_factor", 0.9);

        // Get parameters
        this->get_parameter("position_safety_margin", position_safety_margin_);
        this->get_parameter("velocity_safety_factor", velocity_safety_factor_);
        this->get_parameter("effort_safety_factor", effort_safety_factor_);

        // Initialize joint limits for humanoid robot
        initialize_joint_limits();

        // Create subscribers
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&JointLimitEnforcer::joint_state_callback, this, _1));

        // Create publishers
        safe_joint_cmd_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/safe_joint_commands", 10);

        violation_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/joint_limit_violation", 10);

        RCLCPP_INFO(this->get_logger(), "Joint Limit Enforcer initialized");
    }

private:
    void initialize_joint_limits()
    {
        // Initialize limits for humanoid joints
        joint_limits_["left_hip_yaw"] = {-1.57, 1.57, 5.0, 100.0};
        joint_limits_["left_hip_roll"] = {-0.785, 0.785, 5.0, 100.0};
        joint_limits_["left_hip_pitch"] = {-2.356, 0.785, 5.0, 100.0};
        joint_limits_["left_knee"] = {0.0, 2.356, 5.0, 100.0};
        joint_limits_["left_ankle_pitch"] = {-0.785, 0.785, 3.0, 50.0};
        joint_limits_["left_ankle_roll"] = {-0.524, 0.524, 3.0, 50.0};

        joint_limits_["right_hip_yaw"] = {-1.57, 1.57, 5.0, 100.0};
        joint_limits_["right_hip_roll"] = {-0.785, 0.785, 5.0, 100.0};
        joint_limits_["right_hip_pitch"] = {-2.356, 0.785, 5.0, 100.0};
        joint_limits_["right_knee"] = {0.0, 2.356, 5.0, 100.0};
        joint_limits_["right_ankle_pitch"] = {-0.785, 0.785, 3.0, 50.0};
        joint_limits_["right_ankle_roll"] = {-0.524, 0.524, 3.0, 50.0};

        joint_limits_["torso_yaw"] = {-1.57, 1.57, 3.0, 80.0};

        joint_limits_["left_shoulder_pitch"] = {-2.094, 1.571, 5.0, 50.0};
        joint_limits_["left_shoulder_roll"] = {-0.785, 1.571, 5.0, 50.0};
        joint_limits_["left_shoulder_yaw"] = {-3.14, 3.14, 5.0, 30.0};
        joint_limits_["left_elbow"] = {-2.793, 0.0, 5.0, 30.0};

        joint_limits_["right_shoulder_pitch"] = {-1.571, 2.094, 5.0, 50.0};
        joint_limits_["right_shoulder_roll"] = {-1.571, 0.785, 5.0, 50.0};
        joint_limits_["right_shoulder_yaw"] = {-3.14, 3.14, 5.0, 30.0};
        joint_limits_["right_elbow"] = {0.0, 2.793, 5.0, 30.0};

        joint_limits_["neck_yaw"] = {-1.571, 1.571, 3.0, 10.0};
        joint_limits_["neck_pitch"] = {-0.785, 0.393, 3.0, 10.0};
        joint_limits_["neck_roll"] = {-0.785, 0.785, 3.0, 10.0};
    }

    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        bool violation_detected = false;
        auto safe_msg = std::make_shared<trajectory_msgs::msg::JointTrajectory>();

        // Set up the safe trajectory message
        safe_msg->joint_names = msg->name;
        safe_msg->header.stamp = this->get_clock()->now();
        safe_msg->points.resize(1);
        safe_msg->points[0].time_from_start.sec = 0;
        safe_msg->points[0].time_from_start.nanosec = 50000000; // 50ms

        // Resize trajectory points
        safe_msg->points[0].positions.resize(msg->position.size());
        safe_msg->points[0].velocities.resize(msg->velocity.size());
        safe_msg->points[0].effort.resize(msg->effort.size());

        for (size_t i = 0; i < msg->name.size(); ++i) {
            std::string joint_name = msg->name[i];

            if (joint_limits_.find(joint_name) != joint_limits_.end()) {
                JointLimits limits = joint_limits_[joint_name];

                double current_pos = 0.0;
                double current_vel = 0.0;
                double current_eff = 0.0;

                if (i < msg->position.size()) current_pos = msg->position[i];
                if (i < msg->velocity.size()) current_vel = msg->velocity[i];
                if (i < msg->effort.size()) current_eff = msg->effort[i];

                // Check and enforce position limits
                double safe_pos = std::max(limits.min_position + position_safety_margin_,
                                         std::min(limits.max_position - position_safety_margin_, current_pos));

                // Check and enforce velocity limits
                double safe_vel = std::max(-limits.max_velocity * velocity_safety_factor_,
                                         std::min(limits.max_velocity * velocity_safety_factor_, current_vel));

                // Check and enforce effort limits
                double safe_eff = std::max(-limits.max_effort * effort_safety_factor_,
                                         std::min(limits.max_effort * effort_safety_factor_, current_eff));

                // Check for violations
                if (std::abs(current_pos - safe_pos) > 0.001 ||
                    std::abs(current_vel - safe_vel) > 0.001 ||
                    std::abs(current_eff - safe_eff) > 0.001) {
                    violation_detected = true;
                    RCLCPP_WARN(this->get_logger(),
                        "Joint limit violation detected for %s: pos=%.3f->%.3f, vel=%.3f->%.3f, eff=%.3f->%.3f",
                        joint_name.c_str(), current_pos, safe_pos, current_vel, safe_vel, current_eff, safe_eff);
                }

                // Set safe values
                safe_msg->points[0].positions[i] = safe_pos;
                safe_msg->points[0].velocities[i] = safe_vel;
                safe_msg->points[0].effort[i] = safe_eff;
            } else {
                // If joint not in our limits map, pass through original values
                if (i < msg->position.size()) safe_msg->points[0].positions[i] = msg->position[i];
                if (i < msg->velocity.size()) safe_msg->points[0].velocities[i] = msg->velocity[i];
                if (i < msg->effort.size()) safe_msg->points[0].effort[i] = msg->effort[i];
            }
        }

        // Publish safe commands
        safe_joint_cmd_pub_->publish(*safe_msg);

        // Publish violation status
        auto violation_msg = std_msgs::msg::Bool();
        violation_msg.data = violation_detected;
        violation_pub_->publish(violation_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr safe_joint_cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr violation_pub_;

    std::unordered_map<std::string, JointLimits> joint_limits_;
    double position_safety_margin_;
    double velocity_safety_factor_;
    double effort_safety_factor_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JointLimitEnforcer>());
    rclcpp::shutdown();
    return 0;
}