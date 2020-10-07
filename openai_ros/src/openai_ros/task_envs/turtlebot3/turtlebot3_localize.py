#!/usr/bin/env python3

import rospy
from openai_ros.robot_envs import turtlebot3_env
from gym import spaces
import numpy as np

class TurtleBot3LocalizeEnv(turtlebot3_env.TurtleBot3Env):
    """
        TurtleBot3LocalizeEnv class is an implementation for localization of turtlebot3 task
    """

    def __init__(self):
        """
        Initialize TurtleBot3LocalizeEnv class

        Parameters
        ----------

        """
        super(TurtleBot3LocalizeEnv, self).__init__()

        rospy.loginfo('status: TurtleBot3LocalizeEnv is ready')
