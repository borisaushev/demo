#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <string>
#include <future> // Mandatory for future checking
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

#include "booster/robot/b1/b1_api_const.hpp"
#include "booster/robot/b1/b1_loco_client.hpp"
#include "booster/robot/channel/channel_factory.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

class DistanceClientNode : public rclcpp::Node
{
public:
    DistanceClientNode()
    : Node("distance_client_node")
    {
        client_ = this->create_client<std_srvs::srv::Trigger>("get_distance");

        worker_thread_ = std::thread(&DistanceClientNode::continuous_control_loop, this);
        worker_thread_.detach();
            
        RCLCPP_INFO(this->get_logger(), "C++ Distance Client Loop Node initialized.");
    }

private:
    const float SPEED = 0.2f;
    const float MIN_DISTANCE = 0.5f;
    const float SIDEWAYS_DISTANCE = 0.5f;

    int move(const float vx, const float vy, const float vyaw, const double duration)
    {
        try
        {
            if (duration <= 0.0)
            {
                std::cerr << "Duration must be greater than zero\n";
                return 1;
            }
            if(vy == 0) {
                std::cout << "Moving FORWARD with speed: " << vx << " for " << duration << " sec\n";
            }
            else {
                std::cout << "Moving SIDEWAYS with speed: " << vy << " for " << duration << " sec\n";
            }

            return 1;

            booster::robot::ChannelFactory::Instance()->Init(0, "127.0.0.1");

            booster::robot::b1::B1LocoClient client;
            client.Init();

            const auto finish =
                std::chrono::steady_clock::now() +
                std::chrono::duration<double>(duration);

            while (std::chrono::steady_clock::now() < finish)
            {
                client.MoveCommand(vx, vy, vyaw);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            // Brake the robot to a complete halt when the loop expires
            client.MoveCommand(0.0F, 0.0F, 0.0F);
            return 0;
        }
        catch (const std::exception &error)
        {
            std::cerr << "Invalid arguments: " << error.what() << '\n';
            return 1;
        }
    }


    void continuous_control_loop()
    {
        while (rclcpp::ok()) {
            
            if (!client_->wait_for_service(1s)) {
                RCLCPP_WARN(this->get_logger(), "Distance service offline. Retrying");
                std::this_thread::sleep_for(500ms);
                continue;
            }

            auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
            auto result_future = client_->async_send_request(request);

            std::future_status status = result_future.wait_for(2.5s);

            if (status == std::future_status::ready)
            {
                auto response = result_future.get();
                
                if (response->success) {
                    try {
                        float distance = std::stof(response->message);
                        std::cout << "\n-----------------------------\n";
                        std::cout << "Received Distance: " << distance << "m\n";
                        
                        if (distance > MIN_DISTANCE * 1.5f) {
                            float time = (distance - MIN_DISTANCE) / SPEED;
                            move(SPEED, 0, 0, time); 
                        }
                        else {
                            float time = SIDEWAYS_DISTANCE / SPEED;
                            move(0, SPEED, 0, time);
                        }
                    } 
                    catch (const std::invalid_argument& e) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to parse float from string: %s", response->message.c_str());
                    }
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Service returned success=False (likely point cloud timeout).");
                    std::this_thread::sleep_for(100ms); 
                }
            } else {
                RCLCPP_ERROR(this->get_logger(), "Service call timed out waiting for Python server response.");
            }
            
            std::this_thread::sleep_for(3s);
        }
    }

    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_;
    std::thread worker_thread_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DistanceClientNode>());
    rclcpp::shutdown();
    return 0;
}
