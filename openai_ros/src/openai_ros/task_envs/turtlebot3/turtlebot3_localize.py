#!/usr/bin/env python3

import rospy
from openai_ros.robot_envs import turtlebot3_env
from gym import spaces
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from std_srvs.srv import Empty
from nav_msgs.srv import GetMap
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import dynamic_reconfigure.client as dynamic_reconfig

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.patches import Ellipse
from matplotlib import transforms
import numpy as np

class Map():
    """
        Map class is an implementation to store map details
    """

    def __init__(self):
        """
        Initialize Map class
        """
        super(Map, self).__init__()

        self.origin_x = 0.0
        self.origin_y = 0.0
        self.scale = 0.0

        self.size_x = 0
        self.size_y = 0

        self.grid_cells = None

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

        # code related to motion commands
        self._motion_error = 0.05
        self._update_rate = 30
        self._init_linear_speed = 0.0
        self._init_angular_speed = 0.0
        self._linear_forward_speed = 0.5
        self._linear_turn_speed = 0.05
        self._angular_speed = 0.3

        self._is_new_map = False

        # code realted to sensors
        self._request_map = True
        self._request_laser = True
        self._request_odom = True
        self._request_imu = False
        self._request_amcl = True
        self._request_gazebo_data = True

        # code related to displaying results in matplotlib
        fig = plt.figure(figsize=(6, 6))
        self._plt_ax = fig.add_subplot(111)
        plt.ion()
        plt.show()
        self._map_plt = None
        self._gt_pose_plt = None
        self._gt_heading_plt = None
        self._amcl_pose_plt = None
        self._amcl_heading_plt = None
        self._amcl_confidence_plt = None

        rospy.loginfo('status: TurtleBot3LocalizeEnv is ready')

    def render(self, mode='human'):
        """
        render the output in matplotlib plots
        """

        if self._map_data is not None:
            # environment map
            self._draw_map(self._map_data)
            # groundtruth pose
            self._gt_pose_plt, self._gt_heading_plt = \
                self._draw_robot_pose(self._gazebo_pose,
                                      self._gt_pose_plt,
                                      self._gt_heading_plt, 'blue')
            # amcl pose
            self._amcl_pose_plt, self._amcl_heading_plt = \
                self._draw_robot_pose(self._amcl_pose.pose.pose,
                                      self._amcl_pose_plt,
                                      self._amcl_heading_plt, 'green')
            # amcl pose covariance
            self._amcl_confidence_plt = \
                self._draw_pose_confidence(self._amcl_pose.pose.pose,
                                           self._amcl_pose.pose.covariance,
                                           self._amcl_confidence_plt, 'green')

        plt.draw()
        plt.pause(0.00000000001)

    def _draw_map(self, map):
        """
        Draw environment map

        Parameters
        ----------
        map: Map
            map of environment
        """

        if self._is_new_map:
            x_max = map.size_x/2 -1/(map.scale*2)
            x_min = -x_max
            y_max = map.size_y/2 -1/(map.scale*2)
            y_min = -y_max
            extent = [x_min, x_max, y_min, y_max]
            if self._map_plt == None:
                self._map_plt = self._plt_ax.imshow(map.grid_cells,
                            cmap=plt.cm.binary, origin='lower', extent=extent)

                self._plt_ax.grid()
                self._plt_ax.set_xlim([x_min, x_max])
                self._plt_ax.set_ylim([y_min, y_max])

                ticks_x = np.linspace(x_min, x_max)
                ticks_y = np.linspace(y_min, y_max)
                self._plt_ax.set_xticks(ticks_x, ' ')
                self._plt_ax.set_yticks(ticks_y, ' ')

                self._plt_ax.set_xlabel('x coords')
                self._plt_ax.set_ylabel('y coords')
            else:
                pass

            self._is_new_map = False

    def _draw_robot_pose(self, robot_pose, pose_plt: Wedge, heading_plt, color: str):
        """
        Draw robot pose

        Parameters
        ----------
        robot_pose: geometry_msgs.msg._Pose.Pose
            robot's pose
        pose_plt: matplotlib.patches.Wedge
            plot of robot position
        heading_plt: matplotlib.lines.Line2D
            plot of robot heading
        color: str
            color to render for robot

        Returns
        -------
        pose_plt: matplotlib.patches.Wedge
            updated plot of robot position
        heading_plt: matplotlib.lines.Line2D
            updated plot of robot heading
        """
        if robot_pose is None:
            return

        # rescale robot position
        pose_x = robot_pose.position.x / self._map_data.scale
        pose_y = robot_pose.position.y / self._map_data.scale
        pose_z = robot_pose.position.z / self._map_data.scale
        roll, pitch, yaw = euler_from_quaternion([
                            robot_pose.orientation.x,
                            robot_pose.orientation.y,
                            robot_pose.orientation.z,
                            robot_pose.orientation.w
        ])

        radius = 3
        xdata = [pose_x, pose_x + radius*3*np.cos(yaw)]
        ydata = [pose_y, pose_y + radius*3*np.sin(yaw)]

        if pose_plt == None:
            pose_plt = Wedge((pose_x, pose_y), radius, 0, 360, color=color, alpha=0.5)
            heading_plt, = self._plt_ax.plot(xdata, ydata, color=color, alpha=0.5)
            self._plt_ax.add_artist(pose_plt)
        else:
            pose_plt.update({'center': [pose_x, pose_y]})
            heading_plt.update({'xdata': xdata, 'ydata': ydata})

        return pose_plt, heading_plt

    def _draw_pose_confidence(self, robot_pose, covariance, confidence_plt, color: str, n_std=1.0):
        """
        Draw confidence ellipse around the robot pose

        Parameters
        ----------
        robot_pose: geometry_msgs.msg._Pose.Pose
            robot's pose
        covariance:
            robot's pose covariance
        confidence_plt: matplotlib.patches.Ellipse
            plot of robot position confidence
        color: str
            color to render for confidence
        n_std: float
            number of std to determine ellipse's radius

        Returns
        -------
        confidence_plt: matplotlib.patches.Ellipse
            updated plot of robot position confidence
        """

        # reference  https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
        # cov_xy / np.sqrt(cov_xx * cov_yy)
        pearson = covariance[6*0 + 1]/np.sqrt(covariance[6*0 + 0] * covariance[6*1 + 1])

        # compute eigenvalues and rescale
        ell_radius_x = np.sqrt(1 + pearson) / self._map_data.scale
        ell_radius_y = np.sqrt(1 - pearson) / self._map_data.scale

        # compute mean and std
        scale_x = np.sqrt(covariance[6*0 + 0] / self._map_data.scale) * n_std
        mean_x = robot_pose.position.x / self._map_data.scale
        scale_y = np.sqrt(covariance[6*1 + 1] / self._map_data.scale) * n_std
        mean_y = robot_pose.position.y / self._map_data.scale

        transform = transforms.Affine2D().rotate_deg(45) \
                                         .scale(scale_x, scale_y) \
                                         .translate(mean_x, mean_y)
        if confidence_plt == None:
            confidence_plt = Ellipse((0, 0), width=ell_radius_x, height=ell_radius_y,
                                     facecolor='none', edgecolor=color)
            confidence_plt.set_transform(transform + self._plt_ax.transData)
            self._plt_ax.add_artist(confidence_plt)
        else:
            confidence_plt.width = ell_radius_x
            confidence_plt.height = ell_radius_y
            confidence_plt.set_transform(transform + self._plt_ax.transData)

        return confidence_plt

    def _check_amcl_data_is_ready(self):
        """
        Checks amcl topics are operational
        """
        topic_name = '/particlecloud'
        topic_class = PoseArray
        time_out = 5.0
        self._particle_cloud = self._check_topic_data_is_ready(topic_name, topic_class, time_out)

        topic_name = '/amcl_pose'
        topic_class = PoseWithCovarianceStamped
        time_out = 1.0
        self._amcl_pose = self._check_topic_data_is_ready(topic_name, topic_class, time_out)

    def _check_init_pose_pub_ready(self):
        """
        Checks initial pose publisher is operational
        """
        self._check_publisher_is_ready(self._init_pose_pub)

    def _check_map_data_is_ready(self):
        """
        Checks map service is operational
        """

        service_name = '/static_map'
        service_class = GetMap
        msg = self._call_service(service_name, service_class)

        # initialize map
        map = Map()
        map.scale = msg.map.info.resolution
        map.size_x = msg.map.info.width
        map.size_y = msg.map.info.height

        # rescale the map origin coordinates
        map.origin_x = msg.map.info.origin.position.x + (map.size_x/2) * map.scale
        map.origin_y = msg.map.info.origin.position.y + (map.size_y/2) * map.scale

        map.grid_cells = np.array(msg.map.data).reshape(map.size_y, map.size_x)

        self._map_data = map

        self._is_new_map = True
        self._request_map = False

    def _init_amcl(self, is_global=True):
        """
        Initialize amcl

        Parameters
        ----------
        is_global: bool
            flag to initialize global localization or not
        """

        # publish initialpose for amcl
        init_pose_msg = PoseWithCovarianceStamped()
        init_pose_msg.header.stamp = rospy.get_rostime()
        init_pose_msg.header.frame_id = 'map'

        # position
        init_pose_msg.pose.pose.position.x = 0.0    # pose_x
        init_pose_msg.pose.pose.position.y = 0.0    # pose_y
        init_pose_msg.pose.pose.position.z = 0.0
        # orientation
        quaternion = quaternion_from_euler(0.0, 0.0, 0.0)   # pose_a
        init_pose_msg.pose.pose.orientation.x = quaternion[0]
        init_pose_msg.pose.pose.orientation.y = quaternion[1]
        init_pose_msg.pose.pose.orientation.z = quaternion[2]
        init_pose_msg.pose.pose.orientation.w = quaternion[3]
        # covariance
        covariance = [0.0]*36 # 6x6 covariance
        covariance[6*0 + 0] = 0.5 * 0.5 # cov_xx
        covariance[6*1 + 1] = 0.5 * 0.5 # cov_yy
        covariance[6*5 + 5] = (np.pi/12.0) *(np.pi/12.0)    # cov_aa
        init_pose_msg.pose.covariance = covariance

        self._init_pose_pub.publish(init_pose_msg)

        if is_global:
            # TODO: do we need selective resampling ??

            # dynamic reconfigure
            particles = 20000
            client = dynamic_reconfig.Client('/amcl')
            params = {
                        'max_particles' : particles,
                     }
            config = client.update_configuration(params)

            self._init_global_localization()

        rospy.loginfo('status: amcl initialized')

    def _init_global_localization(self):
        """
        Initialize global localization for amcl
        """

        service_name = '/global_localization'
        service_class = Empty
        self._call_service(service_name, service_class)

    def _set_init_pose(self):
        """
        Set the initial pose of the turtlebot3
        """

        self._move_base( self._init_linear_speed, self._init_angular_speed,
                         self._motion_error, self._update_rate )

        self._init_amcl(is_global=True)

    def _init_env_variables(self):
        """
        Initialize environment variables
        """
        pass

    def _get_obs(self):
        """
        Return the observation from the environment
        """
        pass

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
