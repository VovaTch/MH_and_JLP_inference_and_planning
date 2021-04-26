from __future__ import print_function
import numpy as np
import math
import gtsam


# Camera openess dependent observations
class DAModel:

    def __init__(self, camera_fov_angle_horizontal = math.pi/4,
                 camera_fov_angle_vertical = math.pi/4, range_limit = math.inf, range_minimum = 0):

        # Camera FOV variable in rads
        self.camera_fov_angle_horizontal = camera_fov_angle_horizontal
        self.camera_fov_angle_vertical = camera_fov_angle_vertical
        self.range_limit = range_limit
        self.range_minimum = range_minimum

    # Object observation method. Do we see or not the object? I'm intending to have support for both np.array and pose2
    def object_observation_2d(self, pose_x, pose_o):

        # Turn to list if it is Pose2
        if type(pose_x) is gtsam.Pose2:
            x = []
            x.append(pose_x.x())
            x.append(pose_x.y())
            x.append(pose_x.theta())

        else:
            x = pose_x

        if type(pose_o) is gtsam.Pose2:
            xo = []
            xo.append(pose_o.x())
            xo.append(pose_o.y())
            xo.append(pose_o.theta())
        else:
            xo = pose_o

        # Computing the observation angle and distance from object
        viewpoint = math.atan2((xo[1] - x[1]), (xo[0] - x[0]))
        angle_deviation = self.angle_correction(viewpoint - x[2])
        distance_from_object = self.distance2d(x, xo)

        # Checking if the object is within the FOV
        if abs(angle_deviation) <= self.camera_fov_angle_horizontal and distance_from_object <= self.range_limit:
            return 1
        else:
            return 0

    # Object observation method. Do we see the object or not? Supports Pose3 TODO: check for mathematical soundness
    def object_observation_3d(self, pose_x, pose_o):
        """

        :type pose_x: gtsam.Pose3
        :type pose_o: gtsam.Pose3
        """
        # Extracting translations
        rel_pose = pose_x.between(pose_o)
        relative_yaw = math.atan2(rel_pose.y(), rel_pose.x())
        xy_projection = math.sqrt(rel_pose.y()**2 + rel_pose.x()**2)
        distance_from_object = math.sqrt(rel_pose.x()**2 + rel_pose.y()**2 + rel_pose.z()**2)
        relative_pitch = math.atan2(rel_pose.z(), xy_projection)

        if abs(relative_yaw) <= self.camera_fov_angle_horizontal and \
            abs(relative_pitch) <= self.camera_fov_angle_vertical and \
            distance_from_object <= self.range_limit and \
            distance_from_object >= self.range_minimum:

            return 1
        else:
            return 0

    # Angle correction for range between -pi to pi
    @staticmethod
    def angle_correction(angle):
        while angle <= -math.pi:
            angle += 2 * math.pi
        while angle >= math.pi:
            angle -= 2 * math.pi
        return angle

    # Find distance between two points 2d
    @staticmethod
    def distance2d(point_1, point_2):
        distance = math.sqrt((point_1[0] - point_2[0])**2 +
                             (point_1[1] - point_2[1])**2)
        return distance

    # Find distance between two points 3d
    @staticmethod
    def distance3d(point_1, point_2):
        distance = math.sqrt((point_1[0] - point_2[0]) ** 2 +
                             (point_1[1] - point_2[1]) ** 2 +
                             (point_1[2] - point_2[2]) ** 2)
        return distance