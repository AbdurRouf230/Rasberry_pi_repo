# Source this on BOTH the ground PC and the Raspberry Pi.
# Replace the IP with the ground PC's Ethernet address.

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export ROS_DISCOVERY_SERVER=192.168.10.10:11811
export ROS_LOCALHOST_ONLY=0
