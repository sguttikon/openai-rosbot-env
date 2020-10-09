#!/usr/bin/env python3

import numpy as np
from tf.transformations import euler_from_quaternion

class Map():
    """
        Map class is an implementation to store map details
    """

    def __init__(self):
        """
        Initialize Map class (2D map)
        """
        super(Map, self).__init__()

        self.__origin = Pose()
        self.__scale = 0.05

        self.__width = 0
        self.__height = 0

        self.__grid_cells = None

    def set_origin(self, pose):
        """
        Sets the map origin position

        :param utils.Pose pose: map origin
        """
        self.__origin = pose

    def get_origin(self):
        """
        Gets the map origin position

        :return utils.Pose
        """
        return self.__origin

    def set_scale(self, scale: float):
        """
        Sets the scale (resolution) of map

        :param float scale: The map scale
        """
        self.__scale = scale

    def get_scale(self):
        """
        Gets the scale (resolution) of map

        :return float
        """
        return self.__scale

    def set_size(self, width: int, height: int):
        """
        Sets the width and height of map

        :param int width: The width of map
               int height: The height of map
        """
        self.__width = width
        self.__height = height

    def get_size(self):
        """
        Gets the width and height of map

        :return int, int
        """
        return self.__width, self.__height

    def set_cells(self, cells: tuple):
        """
        Sets the map cells

        :param tuple cells: map grid cells
        """
        self.__grid_cells = np.array(cells).reshape(self.__height, self.__width)

    def get_cells(self):
        """
        Gets the map cells

        :return numpy.ndarray
        """
        return self.__grid_cells

    def __str__(self):
        """
        Override str method of Object

        :return str
        """
        return ' scale: {0:.4f},\n size: ({1}, {2}),\n origin: {3}'.format(self.__scale, self.__width, self.__height, self.__origin.get_position())

class Pose():
    """
        Pose class is an implementation to store pose details
    """

    def __init__(self):
        """
        Initialize Pose class (3D pose)
        """
        super(Pose, self).__init__()

        self.__position = np.zeros(3, dtype=float)
        self.__euler = np.zeros(3)
        self.__quaternion = np.zeros(4)

        self.__covariance = np.zeros((6, 6))
        self.__entropy = 0.0
        self.__error = 0.0

    def set_position(self, x: float, y: float, z:float):
        """
        Sets the 3D position

        :param float x: position in x-axis
               float y: position in y-axis
               float z: position in z-axis
        """
        self.__position[0] = x
        self.__position[1] = y
        self.__position[2] = z

    def get_position(self):
        """
        Gets the 3D position

        :return numpy.ndarray
        """
        return self.__position

    def set_quaternion(self, x: float, y: float, z: float, w: float):
        """
        Sets the quaternion angle

        :param float x: orientation in x
               float y: orientation in y
               float z: orientation in z
               float w: orientation in w
        """
        self.__quaternion[0] = x
        self.__quaternion[1] = y
        self.__quaternion[2] = z
        self.__quaternion[3] = w

        roll, pitch, yaw = euler_from_quaternion([x, y, z, w])
        self.__set_euler(roll, pitch, yaw)

    def get_quaternion(self):
        """
        Gets the quaternion angle

        :return numpy.ndarray
        """
        return self.__quaternion

    def __set_euler(self, roll: float, pitch: float, yaw: float):
        """
        Sets the euler angle

        :param float roll: roll orientation
               float pitch: pitch orientation
               float yaw: yaw orientation
        """
        self.__euler[0] = roll
        self.__euler[1] = pitch
        self.__euler[2] = yaw

    def get_euler(self):
        """
        Gets the euler angle

        :return numpy.ndarray
        """
        return self.__euler

    def set_covariance(self, covariance):
        """
        Sets the pose covariance

        :param tuple covariance: pose covariance
        """
        self.__covariance = np.array(covariance).reshape((6, 6))
        self.__calculate_entropy()

    def get_covariance(self):
        """
        Gets the pose covariance

        :return numpy.ndarray
        """
        return self.__covariance

    def __calculate_entropy(self):
        """
        calculate entropy based on the pose covariance
        """

        mean = np.array([self.__position[0], self.__position[1], self.__euler[2]])
        cov = np.array([
            [self.__covariance[0,0], self.__covariance[0,1], self.__covariance[0,5]],
            [self.__covariance[1,0], self.__covariance[1,1], self.__covariance[1,5]],
            [self.__covariance[5,0], self.__covariance[5,2], self.__covariance[5,5]]
        ])

        # reference https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        entropy = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * cov))

        # TODO: do we need to fix this?
        # for singular matrix, entropy value is infinity
        self.__entropy = entropy

    def get_entropy(self):
        """
        Gets the pose entropy

        :return float
        """
        return self.__entropy

    def set_estimate_error(self, error):
        """
        Sets the squared euclidean error in pose estimate

        :param floag error: squared euclidean error
        """
        self.__error = error

    def get_estimate_error(self):
        """
        Gets the squared euclidean error in pose estimate
        """
        return self.__error

    def __str__(self):
        """
        Override str method of Object

        :return str
        """
        return ' position: {0},\n orientation: {1},\n covariance: {2}'.format(self.__position, self.__euler, self.__covariance)
