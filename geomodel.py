from __future__ import print_function
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, O


# Basic relative pose measurement model
class GeoModel:

    def __init__(self, model_noise):

        # The instances differ from each other by model noise
        self.model_noise = model_noise

        # Symbol initialization
        self.X = lambda i: X(i)
        self.XO = lambda j: O(j)

    def add_measurement(self, measurement, new_da_realization, graph, initial, da_realization):
        """

        :type graph: gtsam.NonlinearFactorGraph
        :type initial: gtsam.Values
        """

        # Adding a factor per each measurement with corresponding data association realization
        measurement_number = 0
        path_length = len(da_realization)
        for realization in new_da_realization:
            realization = int(realization)
            graph.add(gtsam.BetweenFactorPose3(self.X(path_length), self.XO(realization),
                                               measurement[measurement_number], self.model_noise))
            measurement_number += 1

            # Initialization of object of the realization
            # TODO: CHANGE TO SOMETHING MORE ACCURATE / LESS BASIC
            if initial.exists(self.XO(realization)) is False:
                initial.insert(self.XO(realization), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))

        # Saving the data association realization in a list
        da_realization[path_length-1] = new_da_realization

    # Compute un-normlized log-likelihood of measurement
    def log_pdf_value(self, measurement, point_x, point_xo):
        """

        :type point_x: gtsam.Pose3
        :type point_xo: gtsam.Pose3
        :type measurement: gtsam.Pose3
        """

        # Create an auxilary factor and compute his log-pdf value (without the normalizing constant)
        aux_factor = gtsam.BetweenFactorPose3(1, 2, measurement, self.model_noise)
        poses = gtsam.Values()
        poses.insert(1, point_x)
        poses.insert(2, point_xo)

        return aux_factor.error(poses)

    @staticmethod
    def model_sample(pose_x, pose_xo, noise=np.eye(6), ML_sample=False):
        """

        :type pose_x: gtsam.Pose3
        :type pose_xo: gtsam.Pose3
        :type rel_pose: gtsam.Pose3
        """

        rel_pose = pose_x.between(pose_xo)
        rel_pose_vector = [rel_pose.rotation().rpy()[0], rel_pose.rotation().rpy()[1], rel_pose.rotation().rpy()[2],
                           rel_pose.x(), rel_pose.y(), rel_pose.z()]
        if ML_sample is False:

            sampled_rel_pose_vector = np.random.multivariate_normal(rel_pose_vector, noise)
            return gtsam.Pose3(gtsam.Rot3.Ypr(sampled_rel_pose_vector[2], sampled_rel_pose_vector[1],
                                                    sampled_rel_pose_vector[0]),
                               np.array([sampled_rel_pose_vector[3], sampled_rel_pose_vector[4],
                                                       sampled_rel_pose_vector[5]]))
        else:
            return rel_pose

