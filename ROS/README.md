# ROS Commands for Bot

## Laser Avoidance
- roslaunch transbot_laser laser_Avoidance.launch
- rosrun rqt_reconfigure rqt_reconfigure

### Note: Ideally best way to work is connecting to bot remotely (ex. RealVNC)
### You can connect the bot to your own wifi/lan but some nodes may not work (namely rosbridge server) because ROS expects communication to be on 192.168.1.11 IP

## Creating a Map (Do before using RVIS or anything using costmaps):
- roslaunch transbot_nav transbot_bringup.launch
- roslaunch transbot_nav transbot_map.launch map_type:=gmapping
- rosrun teleop_twist_keyboard teleop_twist_keyboard.py **(this is how you move the bot around to create a new global costmap)**
- **Save Map:** rosrun map_server map_saver -f ~/transbot_ws/src/transbot_nav/maps/NAME_OF_MAP

## Running Navigation & Server (use when running ROS Handler / main.py or connecting with Frontend):

### Note: To connect to bot with the ROS handler you MUST be on Transbot Hotspot

- roslaunch transbot_nav transbot_bringup.launch **(might have to run twice on startup as it needs to kill the Transbot startup launch)**
- roslaunch rosbridge_server rosbridge_websocket.launch
- roslaunch transbot_nav transbot_navigation.launch open_rviz:=true map:=NAME_OF_MAP