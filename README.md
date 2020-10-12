# openai-rosbot-env

Trying to create customized gym environments based on ros/gazebo platform


### Setup:
Ros Distribution - [ROS Noetic Ninjemys](http://wiki.ros.org/noetic) <br>
Operating System - Ubuntu 20.04 (Focal)  <br>
Python Distribution - Python 3.X <br>
Gym - [doc](https://gym.openai.com/docs/)

#### Steps: 

- create gazebo environment
  1. convert layout xxx.png to occupancy map xxx.pgm using [layout_to_occpmap.py](https://github.com/suresh-guttikonda/openai-rosbot-env/blob/master/gazebo_models/indoor_layouts/src/layout_to_occpmap.py) <br>
  2. create corresponding xxx.yaml for occupancy map as [sample](https://github.com/suresh-guttikonda/openai-rosbot-env/tree/master/gazebo_models/indoor_layouts/map/sample)
  3. read the map and publish using command === rosrun map_server map_server xxx.yaml ===
  4. run map2gazebo using command === roslaunch map2gazebo map2gazebo.launch export_dir:=/path/to/export_dir === to read published map and create stl/dae based on map layout
  ##### Note: above occupany map is used only to generate dae/stl gazebo environment, for localization task we still use occupancy map generated from APIs like gmapping

### References:

#### codebase
* [openai-ros](http://wiki.ros.org/openai_ros)
* [custom gym env creation guide](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
* [HouseExpo indoor layout dataset](https://github.com/TeaganLi/HouseExpo)
* [Gazebo env from 2D maps](https://github.com/shilohc/map2gazebo)

#### rosbots
* TurtleBot3 [turtle-bot3](https://github.com/ROBOTIS-GIT/turtlebot3) , [turtle-bot3-msgs](https://github.com/ROBOTIS-GIT/turtlebot3_msgs) , [turtle-bot3-simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations) , [turtle-bot3-driver](https://github.com/ROBOTIS-GIT/hls_lfcd_lds_driver)



#### Notes:
* configure #export TURTLEBOT3_MODEL=waffle in .bashrc file
* assumption is all sensor data, poses, etc are in same frame ie. 'map'
* fix for map2gazebo utf encoding error is to change lines open(export_dir + "/map.stl", 'w') => open(export_dir + "/map.stl", 'wb')
