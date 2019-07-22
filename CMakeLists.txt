project(netvlad_tf)
cmake_minimum_required(VERSION 2.8.3)

# find_package(catkin_simple REQUIRED)
# catkin_simple(ALL_DEPS_REQUIRED)

# cs_install()
# cs_export()

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

catkin_python_setup()

add_message_files(DIRECTORY msg
  FILES
  Descriptor.msg
)

add_service_files(DIRECTORY srv
  FILES
  netvlad_result.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

#############
## Install ##
#############

install(PROGRAMS
  scripts/netvlad_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)