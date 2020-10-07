#!/usr/bin/env python3

import rospy
from openai_ros import rosbot_gazebo_env
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class TurtleBot3Env(rosbot_gazebo_env.RosbotGazeboEnv):
    """
        TurtleBot3Env class acts as abstract turtlebot environment template
    """

    def __init__(self):
        """
        Initialize TurtleBot3Env class

        Sensor Topic List:
        * /odom : Odometry readings of the base of the robot
        * /imu  : IMU readings to get relative accelerations and orientations
        * /scan : Laser readings

        Actuator Topic List:
        * /cmd_vel : Move the robot through Twist commands

        """

        super(TurtleBot3Env, self).__init__(reset_type = 'SIMULATION')

        self._laser_scan = None
        self._imu_data = None
        self._odom_data = None
        self._particle_cloud = None
        self._amcl_pose = None

        # setup subscribers and publishers
        self.gazebo.unpause_sim()
        self._check_all_systems_are_ready()

        rospy.Subscriber('/scan', LaserScan, self._laser_scan_callback)
        rospy.Subscriber('/imu', Imu, self._imu_data_callback)
        rospy.Subscriber('/odom', Odometry, self._odom_data_callback)

        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

        self._check_publishers_connection()
        self._check_all_systems_are_ready()
        self.gazebo.pause_sim()
        rospy.loginfo('status: system check passed')

    def _init_amcl(self):
        """
        Initialize amcl
        """
        raise NotImplementedError()

    def get_laser_scan(self):
        """
        Laser Scan Getter
        """
        return self._laser_scan

    def get_imu_data(self):
        """
        Imu data Getter
        """
        return self._imu_data

    def get_odom_data(self):
        """
        Odometry data Getter
        """
        return self._odom_data
    
    def get_amcl_pose(self):
    	"""
    	AMCL pose Getter
    	"""
    	return self._amcl_pose
    
    def get_particle_cloud(self):
    	"""
    	AMCL particle cloud Getter
    	"""
    	return self._particle_cloud

    def _check_all_systems_are_ready(self):
        """
        Checks all sensors and other simulation systems are operational
        """

        self._check_all_sensors_are_ready()


        self._check_amcl_data_is_ready()

    def _check_publishers_connection(self):
        """
        Checks all publishers are operational
        """
        self._check_cmd_vel_pub_ready()

    def _check_cmd_vel_pub_ready(self):
        """
        Checks command velocity is operational
        """
        rate = rospy.Rate(10) # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                rate.sleep()
            except rospy.ROSInterruptException as e:
                # do nothing
                pass

    def _check_all_sensors_are_ready(self):
        """
        Checks all sensors are operational
        """

        self._check_laser_scan_is_ready()
        self._check_imu_data_is_ready()
        self._check_odom_data_is_ready()

    def _check_amcl_data_is_ready(self):
        """
        Checks amcl is operational
        """
        pass

    def _check_laser_scan_is_ready(self):
        """
        Checks laser scanner is operational

        Returns
        -------
        laser_scan: rospy.Message
            Message
        """

        topic_name = '/scan'
        topic_class = LaserScan
        time_out = 1.0
        self._laser_scan  = self._check_sensor_data_is_ready(topic_name, topic_class, time_out)
        return self._laser_scan

    def _check_imu_data_is_ready(self):
        """
        Checks imu is operational

        Returns
        -------
        imu_data: rospy.Message
            Message
        """

        topic_name = '/imu'
        topic_class = Imu
        time_out = 5.0
        self._imu_data = self._check_sensor_data_is_ready(topic_name, topic_class, time_out)
        return self._imu_data

    def _check_odom_data_is_ready(self):
        """
        Checks odom is operational

        Returns
        -------
        odom_data: rospy.Message
            Message
        """

        topic_name = '/odom'
        topic_class = Odometry
        time_out = 5.0
        self._odom_data = self._check_sensor_data_is_ready(topic_name, topic_class, time_out)
        return self._odom_data

    def _check_sensor_data_is_ready(self, topic_name: str, topic_class, time_out: float):
        """
        Check whethe the sensor is operational by
            1. subscribing to topic_name
            2. receive one topic_class message
            3. unsubscribe

        Parameters
        ----------
        topic_name:
            name of the topic
        topic_class:
            topic type
        time_out:
            timeout in seconds

        Returns
        -------
        sensor_data: rospy.Message
            Message

        """

        # TODO: do we need to add max retry limit ??

        sensor_data = None
        # loop until the ros is shutdown or service call is successful
        while sensor_data is None and not rospy.is_shutdown():
            try:
                # create a new subscription to topic, receive one message and then unsubscribe
                sensor_data = rospy.wait_for_message(topic_name, topic_class, timeout = time_out)
            except rospy.ROSException as e:
                # do nothing
                pass

        return sensor_data
    
    def _call_service(self, service_name: str, service_class, max_retry: int = 10):
        """
        Create a service proxy for given service_name and service_class and
        call the service

        Parameters
        ----------
        service_name: str
            name of the service
        service_class:
            service type
        max_retry: int
            maximum number of times to retry calling the service

        """

        # wait until the service becomes available
        rospy.wait_for_service(service_name, timeout=None)
        # create callable proxy to the service
        service_proxy = rospy.ServiceProxy(service_name, service_class)

        is_call_successful = False
        counter = 0

        # loop until the counter reached max retry limit or
        # until the ros is shutdown or service call is successful
        while not is_call_successful and not rospy.is_shutdown():
            if counter < max_retry:
                try:
                    # call service
                    service_proxy()
                    is_call_successful = True
                except rospy.ServiceException as e:
                    # service call failed increment the counter
                    counter += 1
            else:
                # max retry count reached
                rospy.logerr('call to the service %s failed', service_name)
                break

    def _laser_scan_callback(self, data):
        """
        This function is called when laser scan is received

        Parameters
        ----------
        data: rospy.Message
            data received from laser scan topic
        """

        self._laser_scan = data

    def _imu_data_callback(self, data):
        """
        This function is called when imu data is received

        Parameters
        ----------
        data: rospy.Message
            data received from imu topic
        """

        self._imu_data = data

    def _odom_data_callback(self, data):
        """
        This function is called when odom data is received

        Parameters
        ----------
        data: rospy.Message
            data received from odom topic
        """

        self._odom_data = data

    def _move_base(self, linear_speed: float, angular_speed: float,
                    motion_error: float = 0.05, update_rate: float = 10):
        """
        Move the rosbot base, based on the linear and angular speeds.

        Parameters
        ----------
        linear_speed: float
            speed in the linear motion of the robot base frame (x-axis direction)
        angular_speed: float
            speed of the angular turning of the robot base frame
        motion_error: float
            acceptable deviation from the given speed and odometry readings
        update_rate: float
            rate at which we check the odometry

        """

        # publish twist message
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_speed
        cmd_vel_msg.angular.z = angular_speed
        self._check_cmd_vel_pub_ready()
        self._cmd_vel_pub.publish(cmd_vel_msg)

        # wait for the given twist message to be executed correctly
        delta = self._wait_until_twist_achieved(cmd_vel_msg, motion_error, update_rate)

    def _wait_until_twist_achieved(self, cmd_vel_msg: Twist, motion_error: float, update_rate: float):
        """
        Wait for the robot to achieve given cmd_vel, by referencing the odometry velocity readings

        Parameters
        ----------
        cmd_vel_msg: Twist
            velocity in terms of linear and angular parts
        motion_error: float
            acceptable deviation from the given speed and odometry readings
        update_rate: float
            rate at which we check the odometry

        Returns
        -------
        delta: float
            time taken to achieve required twist (in seconds)
        """

        # compute the acceptable ranges for linear and angular velocity
        min_linear_speed = cmd_vel_msg.linear.x - motion_error
        max_linear_speed = cmd_vel_msg.linear.x + motion_error
        min_angular_speed = cmd_vel_msg.angular.z - motion_error
        max_angular_speed = cmd_vel_msg.angular.z + motion_error

        rate = rospy.Rate(update_rate)
        start_time = rospy.get_rostime().to_sec()
        end_time = 0.0

        # loop until the twist is achieved
        while not rospy.is_shutdown():
            current_odom = self._check_odom_data_is_ready()
            odom_linear_vel = current_odom.twist.twist.linear.x
            odom_angular_vel = current_odom.twist.twist.angular.z

            # check whether linear and angular veloctiy are valid
            is_linear_vel_valid = (
                                    (odom_linear_vel <= max_linear_speed) and
                                    (odom_linear_vel > min_linear_speed)
                                  )
            is_angular_vel_valid = (
                                    (odom_angular_vel <= max_angular_speed) and
                                    (odom_angular_vel > min_angular_speed)
                                   )

            if is_linear_vel_valid and is_angular_vel_valid:
                # required twist achieved
                end_time = rospy.get_rostime().to_sec()
                break
            else:
                # otherwise
                rate.sleep()
        return (end_time - start_time)
