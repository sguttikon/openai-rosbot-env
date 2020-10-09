# openai-rosbot-env

Trying to create customized gym environments based on ros/gazebo platform


### Setup:
Ros Distribution - [ROS Noetic Ninjemys](http://wiki.ros.org/noetic) <br>
Operating System - Ubuntu 20.04 (Focal)  <br>
Python Distribution - Python 3.X <br>
Gym - [doc](https://gym.openai.com/docs/)


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
