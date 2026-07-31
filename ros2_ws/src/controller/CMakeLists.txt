cmake_minimum_required(VERSION 3.8)
project(controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)


# Includes the header folder path
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/booster_robotics_sdk/include")

# Explicitly points to the x86_64 directory containing libbooster_robotics_sdk.a
# link_directories("${CMAKE_CURRENT_SOURCE_DIR}/booster_robotics_sdk/lib/aarch64")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/booster_robotics_sdk/lib/x86_64")

add_executable(controller src/walk_sdk.cpp)

# Target dependencies for ROS 2 packages
ament_target_dependencies(controller
  rclcpp
  std_msgs
  std_srvs 
)

# Links the static library (drops the 'lib' prefix and '.a' extension)
target_link_libraries(controller
  booster_robotics_sdk
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  controller
  DESTINATION lib/${PROJECT_NAME}
)
  


ament_package()
