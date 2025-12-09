// humanoid_hardware_interface.hpp
// Header file for humanoid robot hardware interface

#ifndef HUMANOID_HARDWARE_INTERFACE_HPP_
#define HUMANOID_HARDWARE_INTERFACE_HPP_

#include <string>
#include <vector>
#include <signal.h>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace humanoid_hardware_interface
{
    class HumanoidHardwareInterface : public hardware_interface::SystemInterface
    {
    public:
        RCLCPP_SHARED_PTR_DEFINITIONS(HumanoidHardwareInterface);

        // Callback for initialization
        hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo &info) override;

        // Export state interfaces
        std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

        // Export command interfaces
        std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

        // Callback for activation
        hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &previous_state) override;

        // Callback for deactivation
        hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

        // Read data from hardware
        hardware_interface::return_type read(const rclcpp::Time &time, const rclcpp::Duration &period) override;

        // Write data to hardware
        hardware_interface::return_type write(const rclcpp::Time &time, const rclcpp::Duration &period) override;

    private:
        // Store the joint names
        std::vector<std::string> joint_names_;
        std::vector<std::string> sensor_names_;

        // Store the state of the hardware
        std::vector<double> hw_positions_;
        std::vector<double> hw_velocities_;
        std::vector<double> hw_efforts_;

        // Store the commands to the hardware
        std::vector<double> hw_commands_positions_;
        std::vector<double> hw_commands_velocities_;
        std::vector<double> hw_commands_efforts_;

        // Store sensor data
        std::vector<double> hw_imu_orientation_;
        std::vector<double> hw_imu_angular_velocity_;
        std::vector<double> hw_imu_linear_acceleration_;
        std::vector<double> hw_force_torque_;
    };

} // namespace humanoid_hardware_interface

#endif // HUMANOID_HARDWARE_INTERFACE_HPP_