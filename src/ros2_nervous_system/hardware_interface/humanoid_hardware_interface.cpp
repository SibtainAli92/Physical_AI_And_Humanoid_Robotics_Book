// humanoid_hardware_interface.cpp
// Hardware interface for humanoid robot communication with real controllers

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "humanoid_hardware_interface/humanoid_hardware_interface.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace humanoid_hardware_interface
{
    // Callback function to handle real-time signals
    void signal_handler(int sig)
    {
        RCLCPP_INFO(rclcpp::get_logger("humanoid_hardware_interface"), "Received signal %d, shutting down...", sig);
        exit(0);
    }

    CallbackReturn HumanoidHardwareInterface::on_init(const hardware_interface::HardwareInfo &info)
    {
        if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS)
        {
            return CallbackReturn::ERROR;
        }

        // Initialize joint data structures
        hw_positions_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
        hw_velocities_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
        hw_efforts_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
        hw_commands_positions_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
        hw_commands_velocities_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
        hw_commands_efforts_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());

        // Initialize sensor data structures
        hw_imu_orientation_.resize(4, std::numeric_limits<double>::quiet_NaN()); // x, y, z, w
        hw_imu_angular_velocity_.resize(3, std::numeric_limits<double>::quiet_NaN()); // x, y, z
        hw_imu_linear_acceleration_.resize(3, std::numeric_limits<double>::quiet_NaN()); // x, y, z
        hw_force_torque_.resize(6, std::numeric_limits<double>::quiet_NaN()); // force x, y, z and torque x, y, z

        // Initialize joint names
        for (const auto &joint : info_.joints)
        {
            joint_names_.push_back(joint.name);
        }

        // Initialize sensor names
        for (const auto &sensor : info_.sensors)
        {
            sensor_names_.push_back(sensor.name);
        }

        // Setup real-time signal handling
        signal(SIGTERM, signal_handler);
        signal(SIGINT, signal_handler);

        RCLCPP_INFO(rclcpp::get_logger("humanoid_hardware_interface"), "Initialized hardware interface for %zu joints", info_.joints.size());

        return CallbackReturn::SUCCESS;
    }

    std::vector<hardware_interface::StateInterface> HumanoidHardwareInterface::export_state_interfaces()
    {
        std::vector<hardware_interface::StateInterface> state_interfaces;
        for (auto i = 0u; i < info_.joints.size(); i++)
        {
            state_interfaces.emplace_back(hardware_interface::StateInterface(
                info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]));
            state_interfaces.emplace_back(hardware_interface::StateInterface(
                info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]));
            state_interfaces.emplace_back(hardware_interface::StateInterface(
                info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_efforts_[i]));
        }

        // Add IMU sensor interfaces
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "imu_sensor", "orientation.x", &hw_imu_orientation_[0]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "imu_sensor", "orientation.y", &hw_imu_orientation_[1]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "imu_sensor", "orientation.z", &hw_imu_orientation_[2]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "imu_sensor", "orientation.w", &hw_imu_orientation_[3]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "imu_sensor", "angular_velocity.x", &hw_imu_angular_velocity_[0]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "imu_sensor", "angular_velocity.y", &hw_imu_angular_velocity_[1]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "imu_sensor", "angular_velocity.z", &hw_imu_angular_velocity_[2]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "imu_sensor", "linear_acceleration.x", &hw_imu_linear_acceleration_[0]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "imu_sensor", "linear_acceleration.y", &hw_imu_linear_acceleration_[1]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "imu_sensor", "linear_acceleration.z", &hw_imu_linear_acceleration_[2]));

        // Add force/torque sensor interfaces
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "force_torque_sensor", "force.x", &hw_force_torque_[0]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "force_torque_sensor", "force.y", &hw_force_torque_[1]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "force_torque_sensor", "force.z", &hw_force_torque_[2]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "force_torque_sensor", "torque.x", &hw_force_torque_[3]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "force_torque_sensor", "torque.y", &hw_force_torque_[4]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            "force_torque_sensor", "torque.z", &hw_force_torque_[5]));

        return state_interfaces;
    }

    std::vector<hardware_interface::CommandInterface> HumanoidHardwareInterface::export_command_interfaces()
    {
        std::vector<hardware_interface::CommandInterface> command_interfaces;
        for (auto i = 0u; i < info_.joints.size(); i++)
        {
            command_interfaces.emplace_back(hardware_interface::CommandInterface(
                info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_positions_[i]));
            command_interfaces.emplace_back(hardware_interface::CommandInterface(
                info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_commands_velocities_[i]));
            command_interfaces.emplace_back(hardware_interface::CommandInterface(
                info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_commands_efforts_[i]));
        }

        return command_interfaces;
    }

    CallbackReturn HumanoidHardwareInterface::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
    {
        RCLCPP_INFO(rclcpp::get_logger("humanoid_hardware_interface"), "Activating hardware interface...");

        // Initialize all joints to zero position
        for (auto i = 0u; i < hw_positions_.size(); i++)
        {
            if (std::isnan(hw_positions_[i]))
            {
                hw_positions_[i] = 0.0;
                hw_velocities_[i] = 0.0;
                hw_efforts_[i] = 0.0;
            }
        }

        // Initialize all commands to zero
        for (auto i = 0u; i < hw_commands_positions_.size(); i++)
        {
            if (std::isnan(hw_commands_positions_[i]))
            {
                hw_commands_positions_[i] = hw_positions_[i];
                hw_commands_velocities_[i] = 0.0;
                hw_commands_efforts_[i] = 0.0;
            }
        }

        RCLCPP_INFO(rclcpp::get_logger("humanoid_hardware_interface"), "Successfully activated hardware interface");

        return CallbackReturn::SUCCESS;
    }

    CallbackReturn HumanoidHardwareInterface::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
    {
        RCLCPP_INFO(rclcpp::get_logger("humanoid_hardware_interface"), "Deactivating hardware interface...");

        return CallbackReturn::SUCCESS;
    }

    hardware_interface::return_type HumanoidHardwareInterface::read(
        const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
    {
        // Read data from actual hardware
        // In simulation, we'll just copy the command values to state
        // In real hardware, this would involve communication with actuators and sensors

        // For simulation purposes, we'll update positions based on commands
        // In real hardware, this would read actual sensor values
        for (auto i = 0u; i < hw_positions_.size(); i++)
        {
            // In real hardware, read actual position from encoders
            // For simulation, we'll just update based on commands with some dynamics
            if (!std::isnan(hw_commands_positions_[i]))
            {
                // Simple first-order dynamics for simulation
                double position_error = hw_commands_positions_[i] - hw_positions_[i];
                hw_positions_[i] += position_error * 0.1; // 10% of error per cycle
            }

            // Read actual velocities and efforts from hardware
            // For simulation, we'll estimate velocity from position change
            if (i < hw_velocities_.size())
            {
                hw_velocities_[i] = 0.0; // Read from hardware or estimate
            }
            if (i < hw_efforts_.size())
            {
                hw_efforts_[i] = 0.0; // Read from hardware
            }
        }

        // Read IMU data from hardware
        // For simulation, we'll set some default values
        hw_imu_orientation_[0] = 0.0; // x
        hw_imu_orientation_[1] = 0.0; // y
        hw_imu_orientation_[2] = 0.0; // z
        hw_imu_orientation_[3] = 1.0; // w
        hw_imu_angular_velocity_[0] = 0.0; // x
        hw_imu_angular_velocity_[1] = 0.0; // y
        hw_imu_angular_velocity_[2] = 0.0; // z
        hw_imu_linear_acceleration_[0] = 0.0; // x
        hw_imu_linear_acceleration_[1] = 0.0; // y
        hw_imu_linear_acceleration_[2] = 9.81; // z (gravity)

        // Read force/torque data from hardware
        hw_force_torque_[0] = 0.0; // force x
        hw_force_torque_[1] = 0.0; // force y
        hw_force_torque_[2] = 0.0; // force z
        hw_force_torque_[3] = 0.0; // torque x
        hw_force_torque_[4] = 0.0; // torque y
        hw_force_torque_[5] = 0.0; // torque z

        return hardware_interface::return_type::OK;
    }

    hardware_interface::return_type HumanoidHardwareInterface::write(
        const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
    {
        // Write commands to actual hardware
        // In simulation, we'll just copy command values
        // In real hardware, this would involve sending commands to actuators

        // In real hardware, this would send commands to the actuators
        // For now, we'll just log the commands being sent
        for (auto i = 0u; i < hw_commands_positions_.size(); i++)
        {
            if (!std::isnan(hw_commands_positions_[i]))
            {
                // Send position command to actuator i
                // This would involve actual hardware communication in real implementation
            }
        }

        return hardware_interface::return_type::OK;
    }

} // namespace humanoid_hardware_interface

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
    humanoid_hardware_interface::HumanoidHardwareInterface,
    hardware_interface::SystemInterface)