#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "booster/robot/b1/b1_api_const.hpp"
#include "booster/robot/b1/b1_loco_client.hpp"
#include "booster/robot/channel/channel_factory.hpp"

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
using std::placeholders::_1;

int move(const float vx, const float vy, const float vyaw, const double duration)
{
    try
    {
        if (duration <= 0.0)
        {
            std::cerr << "Duration must be greater than zero\n";
            return 1;
        }

        booster::robot::ChannelFactory::Instance()->Init(
            0,
            "127.0.0.1");

        booster::robot::b1::B1LocoClient client;
        client.Init();

        // const int mode_result = client.ChangeMode(
        // booster::robot::RobotMode::kWalking);

        // if (mode_result != 0)
        // {
        //     std::cerr << "Failed to enable walking mode: "
        //               << mode_result << '\n';
        //     return 1;
        // }

        const auto finish =
            std::chrono::steady_clock::now() +
            std::chrono::duration<double>(duration);

        while (std::chrono::steady_clock::now() < finish)
        {
            client.MoveCommand(vx, vy, vyaw);
            std::this_thread::sleep_for(
                std::chrono::milliseconds(50));
        }

        client.MoveCommand(0.0F, 0.0F, 0.0F);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Invalid arguments: " << error.what() << '\n';
        return 1;
    }
}

class DistanceSubscriber : public rclcpp::Node
{
public:
    DistanceSubscriber()
    : Node("distance_subscriber")
    {
    subscription_ = this->create_subscription<std_msgs::msg::Float32>(
        "distance_publisher", 1, std::bind(&DistanceSubscriber::topic_callback, this, _1));
    }
    
private:
    const float SPEED = 0.2;
    const float MIN_DISTANCE = 0.5;
    const float SIDEWAYS_DISTANCE = 0.5;

    void topic_callback(const std_msgs::msg::Float32 &msg) const
    {
        float distance = msg.data;
        std::cout << "Distance: " << distance << '\n';
        if(distance > MIN_DISTANCE * 1.5) {
            std::cout << "Moving forward\n";
            float time = (distance - MIN_DISTANCE) / SPEED;
            move(SPEED, 0, 0, time);
        }
        else {
            std::cout << "Moving sideways\n";
            float time = SIDEWAYS_DISTANCE / SPEED;
            move(0, SPEED, 0, time);
        }
    }
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DistanceSubscriber>());
    rclcpp::shutdown();
    return 0;
}