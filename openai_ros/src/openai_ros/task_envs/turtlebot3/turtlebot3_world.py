#!/usr/bin/env python3

import rospy
from openai_ros.robot_envs import turtlebot3_env
from gym import spaces
import numpy as np

class TurtleBot3WorldEnv(turtlebot3_env.TurtleBot3Env):
    """
        TurtleBot3WorldEnv class is an implementation for general turtlebot3 task
    """

    def __init__(self, name_space: str = 'turtlebot3'):
        """
        Initialize TurtleBot3WorldEnv class

        Parameters
        ----------
        name_space: str
            string used to uniquely identify the ros node and related parameters
            refer (turtlebot3_params.yaml)

        """
        super(TurtleBot3WorldEnv, self).__init__()

        self._num_actions = rospy.get_param('/' + name_space + '/n_actions')
        self._skip_beam_interval = rospy.get_param('/' + name_space + '/skip_beam_interval')
        self._min_laser_value = rospy.get_param('/' + name_space + '/min_laser_value')
        self._max_laser_value = rospy.get_param('/' + name_space + '/max_laser_value')
        self._init_linear_speed = rospy.get_param('/' + name_space + '/init_linear_speed')
        self._init_angular_speed = rospy.get_param('/' + name_space + '/init_angular_speed')
        self._linear_forward_speed = rospy.get_param('/' + name_space + '/linear_forward_speed')
        self._linear_turn_speed = rospy.get_param('/' + name_space + '/linear_turn_speed')
        self._angular_speed = rospy.get_param('/' + name_space + '/angular_speed')

        # construct observation space
        laser_scan = self.get_laser_scan() # by this point we already executed _check_laser_scan_is_ready()
        num_laser_readings = len(laser_scan.ranges) / self._skip_beam_interval
        high = np.full( int(num_laser_readings), self._max_laser_value , dtype = np.float32)
        low  = np.full( int(num_laser_readings), self._min_laser_value , dtype = np.float32)
        self._observation_space = spaces.Box(low, high)

        # construct action space
        self._action_space = spaces.Discrete(self._num_actions)
        # construct reward range
        self._reward_range = (-np.inf, np.inf)

        self._episode_done = False
        self._motion_error = 0.05
        self._update_rate = 30

        rospy.loginfo('status: TurtleBot3WorldEnv is ready')

    def _set_init_pose(self):
        """
        Set the initial pose of the turtlebot3

        """

        self._move_base( self._init_linear_speed, self._init_angular_speed,
                         self._motion_error, self._update_rate )

    def _init_env_variables(self):
        """
        Initialize environment variables

        """

        self._episode_done = False

    def _set_action(self, action: int):
        """
        Apply the give action to the environment

        Parameters
        ----------
        action: int
            based on the action id number corresponding linear and angular speed for the rosbot is set

        Action List:
        * 0 = MoveFoward
        * 1 = TurnLeft
        * 2 = TurnRight

        """

        if action == 0:     # move forward
            linear_speed = self._linear_forward_speed
            angular_speed = 0.0
        elif action == 1:   # turn left
            linear_speed = self._linear_turn_speed
            angular_speed = self._angular_speed
        elif action == 2:   # turn right
            linear_speed = self._linear_turn_speed
            angular_speed = -1 * self._angular_speed
        else:               # do nothing / stop
            linear_speed = 0.0
            angular_speed = 0.0

        self._move_base( linear_speed, angular_speed,
                         self._motion_error, self._update_rate )

    def _get_obs(self):
        """
        Return the observation from the environment

        """

        laser_scan = self.get_laser_scan()

        # discretize lazer scan
        disc_laser_scan = []
        num_laser_readings = len(laser_scan.ranges) / self._skip_beam_interval
        for i, beam in enumerate(laser_scan.ranges):
            if (i%num_laser_readings == 0):

                if np.isinf(beam):
                    disc_laser_scan.append(self._max_laser_value)
                elif np.isnan(beam):
                    disc_laser_scan.append(self._min_laser_value)
                else:
                    disc_laser_scan.append(beam)

        return disc_laser_scan

    def _is_done(self):
        """
        Indicates whether or not the episode is done

        """

        # TODO
        pass

    def _compute_reward(self, observation, done):
        """
        Calculate the reward based on the observation

        """

        # TODO
        return 0
