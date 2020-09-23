#!/usr/bin/env python3

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)

# set programatically the path to 'openai_ros' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/openai-rosbot-env/openai_ros/src')
from openai_ros.task_envs.turtlebot3 import turtlebot3_world

import gym
import rospy
import numpy as np

if __name__ == '__main__':

    # create a new ros node
    rospy.init_node('turtlebot3_world')

    # create a new gym environment
    env = gym.make('TurtleBot3World-v0')

    observation = env.reset()

    for i in range(10):
        action = np.random.randint(0, 2)
        print(action)
        env.step(action)

    env.close()

    # prevent the code from exiting until an shutdown signal (ctrl+c) is received
    rospy.spin()
