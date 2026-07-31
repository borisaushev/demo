#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <bitbots_msgs/msg/joint_command.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <robocup_mix_interfaces/msg/kick_reference.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Quaternion.hpp>

#include <booster/common/dds/dds_callback.hpp>
#include <booster/idl/b1/LowState.h>
#include <booster/robot/b1/b1_api_const.hpp>
#include <booster/robot/b1/b1_loco_client.hpp>
#include <booster/robot/channel/channel_factory.hpp>
#include <booster/robot/channel/channel_subscriber.hpp>

#include "sdk_kick_publisher.hpp"

using namespace std::chrono_literals;

namespace
{
const std::vector<std::string> kK1JointNames = {
  "AAHead_yaw",
  "Head_pitch",
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
  "Left_Hip_Pitch",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Left_Knee_Pitch",
  "Left_Ankle_Pitch",
  "Left_Ankle_Roll",
  "Right_Hip_Pitch",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Right_Knee_Pitch",
  "Right_Ankle_Pitch",
  "Right_Ankle_Roll",
};

rclcpp::Time sdk_message_time(const rclcpp::Clock::SharedPtr & fallback_clock)
{
  const auto local_now = fallback_clock->now();
  const auto * context = booster::common::GetCurrentDdsMessageContext();
  if (context != nullptr) {
    const auto usable = [&local_now](const std::chrono::system_clock::time_point & point) {
        const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
          point.time_since_epoch()).count();
        if (nanoseconds <= 0) {
          return rclcpp::Time(0, 0, RCL_SYSTEM_TIME);
        }
        const rclcpp::Time candidate(nanoseconds, RCL_SYSTEM_TIME);
        // Some firmware versions expose a source clock that is several seconds
        // away from the computer clock.  Such a stamp cannot be combined with
        // ZED/ROS TF, so prefer the local DDS receive time in that case.
        const double clock_skew_seconds = std::abs(
          static_cast<double>(nanoseconds - local_now.nanoseconds()) / 1.0e9);
        if (clock_skew_seconds > 1.0) {
          return rclcpp::Time(0, 0, RCL_SYSTEM_TIME);
        }
        return candidate;
      };

    if (context->has_source_timestamp) {
      const auto source = usable(context->source_timestamp_system_time);
      if (source.nanoseconds() > 0) {
        return source;
      }
    }
    if (context->has_reception_timestamp) {
      const auto reception = usable(context->reception_timestamp_system_time);
      if (reception.nanoseconds() > 0) {
        return reception;
      }
    }
    const auto dds_rx = usable(context->dds_rx_system_time);
    if (dds_rx.nanoseconds() > 0) {
      return dds_rx;
    }
  }
  return local_now;
}
}  // namespace

class BoosterSdkBridge : public rclcpp::Node
{
public:
  BoosterSdkBridge()
  : Node("booster_sdk_bridge"),
    tare_pending_(true),
    zero_rotation_(0.0, 0.0, 0.0, 1.0),
    visual_kick_state_(VisualKickState::kIdle),
    have_kick_reference_(false),
    have_cmd_vel_(false)
  {
    declare_parameter<std::string>("network_interface", "127.0.0.1");
    declare_parameter<bool>("tare_imu_on_start", true);
    declare_parameter<int>("visual_kick.version", 0);
    declare_parameter<double>("visual_kick.deceleration_seconds", 0.5);
    declare_parameter<double>("visual_kick.duration_seconds", 7.0);
    declare_parameter<double>("visual_kick.reference_frequency", 50.0);
    declare_parameter<double>("visual_kick.max_reference_age_seconds", 0.5);
    declare_parameter<double>("velocity_limits.x", 1.0);
    declare_parameter<double>("velocity_limits.y", 0.4);
    declare_parameter<double>("velocity_limits.yaw", 1.2);
    declare_parameter<double>("cmd_vel_timeout_seconds", 0.5);
    declare_parameter<double>("head_limits.pitch_min", -0.349);
    declare_parameter<double>("head_limits.pitch_max", 0.855);
    declare_parameter<double>("head_limits.yaw_min", -1.0);
    declare_parameter<double>("head_limits.yaw_max", 1.0);
    declare_parameter<std::vector<std::string>>(
      "joint_names", kK1JointNames);

    network_interface_ = get_parameter("network_interface").as_string();
    tare_pending_ = get_parameter("tare_imu_on_start").as_bool();
    visual_kick_version_ = get_parameter("visual_kick.version").as_int();
    visual_kick_deceleration_ = get_parameter("visual_kick.deceleration_seconds").as_double();
    visual_kick_duration_ = get_parameter("visual_kick.duration_seconds").as_double();
    visual_kick_max_reference_age_ =
      get_parameter("visual_kick.max_reference_age_seconds").as_double();
    velocity_limit_x_ = std::abs(get_parameter("velocity_limits.x").as_double());
    velocity_limit_y_ = std::abs(get_parameter("velocity_limits.y").as_double());
    velocity_limit_yaw_ = std::abs(get_parameter("velocity_limits.yaw").as_double());
    cmd_vel_timeout_ = std::max(0.05, get_parameter("cmd_vel_timeout_seconds").as_double());
    head_pitch_min_ = get_parameter("head_limits.pitch_min").as_double();
    head_pitch_max_ = get_parameter("head_limits.pitch_max").as_double();
    head_yaw_min_ = get_parameter("head_limits.yaw_min").as_double();
    head_yaw_max_ = get_parameter("head_limits.yaw_max").as_double();
    joint_names_ = get_parameter("joint_names").as_string_array();

    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 1);
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("imu/data", 1);

    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 1,
      [this](const geometry_msgs::msg::Twist & msg) {on_cmd_vel(msg);});
    head_command_sub_ = create_subscription<bitbots_msgs::msg::JointCommand>(
      "joint_command/head", 1,
      [this](const bitbots_msgs::msg::JointCommand & msg) {on_head_command(msg);});
    kick_reference_sub_ =
      create_subscription<robocup_mix_interfaces::msg::KickReference>(
      "kick_reference", 1,
      [this](const robocup_mix_interfaces::msg::KickReference & msg) {
        std::lock_guard<std::mutex> lock(kick_reference_mutex_);
        kick_reference_ = msg;
        have_kick_reference_ = true;
      });
    visual_kick_sub_ = create_subscription<std_msgs::msg::Bool>(
      "visual_kick", 1,
      [this](const std_msgs::msg::Bool & msg) {on_visual_kick_request(msg.data);});

    get_up_service_ = create_service<std_srvs::srv::Trigger>(
      "get_up",
      [this](
        const std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr response)
      {
        const int result = loco_client_->GetUp();
        response->success = result == 0;
        response->message = "Booster SDK GetUp result=" + std::to_string(result);
      });
    walking_service_ = create_service<std_srvs::srv::Trigger>(
      "walking_mode",
      [this](
        const std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr response)
      {
        const int result = loco_client_->ChangeMode(booster::robot::RobotMode::kWalking);
        response->success = result == 0;
        response->message = "Booster SDK walking mode result=" + std::to_string(result);
      });
    tare_service_ = create_service<std_srvs::srv::Trigger>(
      "tare_imu",
      [this](
        const std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr response)
      {
        tare_pending_ = true;
        response->success = true;
        response->message = "IMU tare requested";
      });

    booster::robot::ChannelFactory::Instance()->Init(0, network_interface_);
    loco_client_ = std::make_unique<booster::robot::b1::B1LocoClient>();
    loco_client_->Init();

    sdk_kick_publisher_ = std::make_unique<SdkKickPublisher>();

    booster::robot::ChannelSubscriberOptions low_state_options;
    low_state_options.executor_options.queue_capacity = 8;
    low_state_options.executor_options.overflow_policy =
      booster::robot::ChannelSubscriberOverflowPolicy::kDropOldest;
    low_state_options.executor_options.dispatch_mode =
      booster::common::DdsExecutorDispatchMode::kDedicated;
    low_state_subscriber_ = std::make_unique<
      booster::robot::ChannelSubscriber<booster_interface::msg::LowState>>(
      booster::robot::b1::kTopicLowState,
      [this](const void * message) {on_low_state(message);},
      low_state_options);
    low_state_subscriber_->InitChannel();

    const double requested_frequency =
      std::max(1.0, get_parameter("visual_kick.reference_frequency").as_double());
    kick_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / requested_frequency),
      [this]() {visual_kick_tick();});
    cmd_vel_watchdog_timer_ = create_wall_timer(
      50ms, [this]() {cmd_vel_watchdog_tick();});
    low_state_watchdog_timer_ = create_wall_timer(
      1s, [this]() {low_state_watchdog_tick();});

    RCLCPP_INFO(
      get_logger(),
      "Booster SDK bridge initialized on interface '%s'", network_interface_.c_str());
  }

  ~BoosterSdkBridge() override
  {
    if (visual_kick_state_ != VisualKickState::kIdle && loco_client_) {
      stop_visual_kick();
    }
    if (low_state_subscriber_) {
      low_state_subscriber_->CloseChannel();
    }
    if (sdk_kick_publisher_) {
      sdk_kick_publisher_->close();
    }
    if (loco_client_) {
      loco_client_->MoveCommand(0.0F, 0.0F, 0.0F);
    }
  }

private:
  enum class VisualKickState {kIdle, kDecelerating, kActive};

  void on_cmd_vel(const geometry_msgs::msg::Twist & msg)
  {
    if (visual_kick_state_ != VisualKickState::kIdle) {
      loco_client_->MoveCommand(0.0F, 0.0F, 0.0F);
      return;
    }
    if (!std::isfinite(msg.linear.x) || !std::isfinite(msg.linear.y) ||
      !std::isfinite(msg.angular.z))
    {
      RCLCPP_ERROR(get_logger(), "Rejected non-finite cmd_vel");
      loco_client_->MoveCommand(0.0F, 0.0F, 0.0F);
      have_cmd_vel_ = false;
      return;
    }
    last_cmd_vel_time_ = now();
    have_cmd_vel_ = true;
    loco_client_->MoveCommand(
      static_cast<float>(std::clamp(msg.linear.x, -velocity_limit_x_, velocity_limit_x_)),
      static_cast<float>(std::clamp(msg.linear.y, -velocity_limit_y_, velocity_limit_y_)),
      static_cast<float>(std::clamp(msg.angular.z, -velocity_limit_yaw_, velocity_limit_yaw_)));
  }

  void cmd_vel_watchdog_tick()
  {
    if (!have_cmd_vel_ || visual_kick_state_ != VisualKickState::kIdle) {
      return;
    }
    if ((now() - last_cmd_vel_time_).seconds() <= cmd_vel_timeout_) {
      return;
    }
    loco_client_->MoveCommand(0.0F, 0.0F, 0.0F);
    have_cmd_vel_ = false;
    RCLCPP_WARN(get_logger(), "cmd_vel timeout; sent zero velocity");
  }

  void low_state_watchdog_tick()
  {
    if (!low_state_subscriber_) {
      return;
    }

    const auto received = low_state_message_count_.load(std::memory_order_relaxed);
    const auto matched = low_state_subscriber_->GetMatchedPublicationsCount();
    if (received == last_reported_low_state_count_) {
      RCLCPP_WARN(
        get_logger(),
        "No rt/low_state callback in the last second: matched_publishers=%zu, "
        "callbacks_total=%llu",
        matched,
        static_cast<unsigned long long>(received));
    }
    last_reported_low_state_count_ = received;
  }

  void on_head_command(const bitbots_msgs::msg::JointCommand & msg)
  {
    bool have_yaw = false;
    bool have_pitch = false;
    float yaw = 0.0F;
    float pitch = 0.0F;
    const auto count = std::min(msg.joint_names.size(), msg.positions.size());
    for (std::size_t index = 0; index < count; ++index) {
      if (msg.joint_names[index] == "AAHead_yaw") {
        yaw = static_cast<float>(msg.positions[index]);
        have_yaw = true;
      } else if (msg.joint_names[index] == "Head_pitch") {
        pitch = static_cast<float>(msg.positions[index]);
        have_pitch = true;
      }
    }
    if (!have_yaw || !have_pitch) {
      RCLCPP_WARN(get_logger(), "Head command does not contain both K1 head joints");
      return;
    }
    if (!std::isfinite(yaw) || !std::isfinite(pitch)) {
      RCLCPP_ERROR(get_logger(), "Rejected non-finite head command");
      return;
    }
    pitch = std::clamp(
      pitch, static_cast<float>(head_pitch_min_), static_cast<float>(head_pitch_max_));
    yaw = std::clamp(
      yaw, static_cast<float>(head_yaw_min_), static_cast<float>(head_yaw_max_));
    const int result = loco_client_->RotateHead(pitch, yaw);
    if (result != 0) {
      RCLCPP_WARN(get_logger(), "RotateHead failed: %d", result);
    }
  }

  void on_visual_kick_request(bool enabled)
  {
    if (!enabled) {
      if (visual_kick_state_ != VisualKickState::kIdle) {
        stop_visual_kick();
      }
      return;
    }
    if (visual_kick_state_ != VisualKickState::kIdle) {
      return;
    }
    visual_kick_state_ = VisualKickState::kDecelerating;
    visual_kick_phase_start_ = now();
    loco_client_->MoveCommand(0.0F, 0.0F, 0.0F);
    RCLCPP_INFO(get_logger(), "VisualKick requested; decelerating");
  }

  void visual_kick_tick()
  {
    if (visual_kick_state_ == VisualKickState::kIdle) {
      return;
    }

    const bool valid_reference = publish_sdk_kick_reference();
    const double elapsed = (now() - visual_kick_phase_start_).seconds();

    if (visual_kick_state_ == VisualKickState::kDecelerating) {
      loco_client_->MoveCommand(0.0F, 0.0F, 0.0F);
      if (elapsed >= visual_kick_deceleration_) {
        if (!valid_reference) {
          return;
        }
        const auto version = visual_kick_version_ == 1 ?
          booster::robot::b1::VisualKickVersion::kV2 :
          booster::robot::b1::VisualKickVersion::kV1;
        const int result = loco_client_->VisualKick(true, version);
        if (result != 0) {
          RCLCPP_ERROR(get_logger(), "VisualKick start failed: %d", result);
          stop_visual_kick();
          return;
        }
        visual_kick_state_ = VisualKickState::kActive;
        visual_kick_phase_start_ = now();
        RCLCPP_INFO(get_logger(), "VisualKick active (version=%ld)", visual_kick_version_);
      }
      return;
    }

    if (!valid_reference) {
      RCLCPP_ERROR(get_logger(), "VisualKick stopped: kick reference is unavailable or stale");
      stop_visual_kick();
      return;
    }

    if (elapsed >= visual_kick_duration_) {
      stop_visual_kick();
    }
  }

  void stop_visual_kick()
  {
    const auto version = visual_kick_version_ == 1 ?
      booster::robot::b1::VisualKickVersion::kV2 :
      booster::robot::b1::VisualKickVersion::kV1;
    loco_client_->VisualKick(false, version);
    loco_client_->ChangeMode(booster::robot::RobotMode::kWalking);
    visual_kick_state_ = VisualKickState::kIdle;
    RCLCPP_INFO(get_logger(), "VisualKick stopped; walking mode restored");
  }

  bool publish_sdk_kick_reference()
  {
    robocup_mix_interfaces::msg::KickReference reference;
    {
      std::lock_guard<std::mutex> lock(kick_reference_mutex_);
      if (!have_kick_reference_) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "VisualKick is waiting for kick_reference");
        return false;
      }
      reference = kick_reference_;
    }

    const auto current_time = now();
    const rclcpp::Time reference_time(
      reference.header.stamp, current_time.get_clock_type());
    const double reference_age = (current_time - reference_time).seconds();
    if (reference_time.nanoseconds() == 0 ||
      reference_age > visual_kick_max_reference_age_ || reference_age < -0.1)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Ignoring stale kick_reference (age=%.3f s)", reference_age);
      return false;
    }

    const std::array<double, 7> values{
      reference.x, reference.y, reference.dir, reference.goal_x,
      reference.goal_y, reference.robot_theta_to_field, reference.power};
    if (!std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
      }))
    {
      RCLCPP_ERROR(get_logger(), "Ignoring non-finite kick_reference");
      return false;
    }

    SdkKickReference sdk_reference;
    sdk_reference.x = reference.x;
    sdk_reference.y = reference.y;
    sdk_reference.dir = reference.dir;
    sdk_reference.goal_x = reference.goal_x;
    sdk_reference.goal_y = reference.goal_y;
    sdk_reference.robot_theta_to_field = reference.robot_theta_to_field;
    sdk_reference.power = reference.power;
    if (!sdk_kick_publisher_->write(sdk_reference)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to publish SDK kick reference");
      return false;
    }
    return true;
  }

  void on_low_state(const void * message)
  {
    const auto * state = static_cast<const booster_interface::msg::LowState *>(message);
    if (state == nullptr) {
      return;
    }
    low_state_message_count_.fetch_add(1, std::memory_order_relaxed);

    const auto stamp = sdk_message_time(get_clock());
    sensor_msgs::msg::JointState joint_state;
    joint_state.header.stamp = stamp;
    joint_state.header.frame_id = "base";

    // Current K1 firmware publishes URDF joint-space values in the parallel
    // array.  Older firmware may populate only the serial array, so retain it
    // as a compatibility fallback instead of binding TF to one firmware form.
    const auto & parallel_motors = state->motor_state_parallel();
    const auto & serial_motors = state->motor_state_serial();
    const auto & motors = parallel_motors.empty() ? serial_motors : parallel_motors;
    if (!motors.empty()) {
      const auto count = std::min(joint_names_.size(), motors.size());
      joint_state.name.reserve(count);
      joint_state.position.reserve(count);
      joint_state.velocity.reserve(count);
      joint_state.effort.reserve(count);
      for (std::size_t index = 0; index < count; ++index) {
        joint_state.name.push_back(joint_names_[index]);
        joint_state.position.push_back(motors[index].q());
        joint_state.velocity.push_back(motors[index].dq());
        joint_state.effort.push_back(motors[index].tau_est());
      }
      joint_state_pub_->publish(joint_state);
      if (motors.size() != joint_names_.size()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "LowState motor count (%zu) differs from configured joint count (%zu)",
          motors.size(), joint_names_.size());
      }
    } else {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "LowState contains neither parallel nor serial motor states");
    }

    const auto & imu = state->imu_state();
    tf2::Quaternion raw;
    raw.setRPY(imu.rpy()[0], imu.rpy()[1], imu.rpy()[2]);
    if (tare_pending_) {
      zero_rotation_ = raw.inverse();
      tare_pending_ = false;
      RCLCPP_INFO(get_logger(), "Booster IMU tared");
    }
    tf2::Quaternion orientation = zero_rotation_ * raw;
    orientation.normalize();

    sensor_msgs::msg::Imu imu_message;
    imu_message.header.stamp = stamp;
    imu_message.header.frame_id = "imu";
    imu_message.orientation.x = orientation.x();
    imu_message.orientation.y = orientation.y();
    imu_message.orientation.z = orientation.z();
    imu_message.orientation.w = orientation.w();
    imu_message.angular_velocity.x = imu.gyro()[0];
    imu_message.angular_velocity.y = imu.gyro()[1];
    imu_message.angular_velocity.z = imu.gyro()[2];
    imu_message.linear_acceleration.x = imu.acc()[0];
    imu_message.linear_acceleration.y = imu.acc()[1];
    imu_message.linear_acceleration.z = imu.acc()[2];
    imu_pub_->publish(imu_message);
  }

  std::string network_interface_;
  std::vector<std::string> joint_names_;
  std::atomic_bool tare_pending_;
  tf2::Quaternion zero_rotation_;

  long visual_kick_version_;
  double visual_kick_deceleration_;
  double visual_kick_duration_;
  double visual_kick_max_reference_age_;
  double velocity_limit_x_;
  double velocity_limit_y_;
  double velocity_limit_yaw_;
  double cmd_vel_timeout_;
  double head_pitch_min_;
  double head_pitch_max_;
  double head_yaw_min_;
  double head_yaw_max_;
  VisualKickState visual_kick_state_;
  rclcpp::Time visual_kick_phase_start_;

  std::mutex kick_reference_mutex_;
  bool have_kick_reference_;
  robocup_mix_interfaces::msg::KickReference kick_reference_;
  bool have_cmd_vel_;
  rclcpp::Time last_cmd_vel_time_;

  std::unique_ptr<booster::robot::b1::B1LocoClient> loco_client_;
  std::unique_ptr<SdkKickPublisher> sdk_kick_publisher_;
  std::unique_ptr<booster::robot::ChannelSubscriber<booster_interface::msg::LowState>>
    low_state_subscriber_;
  std::atomic<std::uint64_t> low_state_message_count_{0};
  std::uint64_t last_reported_low_state_count_{0};

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<bitbots_msgs::msg::JointCommand>::SharedPtr head_command_sub_;
  rclcpp::Subscription<robocup_mix_interfaces::msg::KickReference>::SharedPtr
    kick_reference_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr visual_kick_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr get_up_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr walking_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr tare_service_;
  rclcpp::TimerBase::SharedPtr kick_timer_;
  rclcpp::TimerBase::SharedPtr cmd_vel_watchdog_timer_;
  rclcpp::TimerBase::SharedPtr low_state_watchdog_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BoosterSdkBridge>());
  rclcpp::shutdown();
  return 0;
}
