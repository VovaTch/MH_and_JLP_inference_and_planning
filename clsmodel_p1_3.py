from __future__ import print_function
import gtsam
import numpy as np
# Fake classifier model import
import libNewFactorFake_p1_3

class ClsModel:

    def __init__(self):
        # Initializing the classifier models, one per class
        self.ClsModelCore = [libNewFactorFake_p1_3.ClsModelFake1(), libNewFactorFake_p1_3.ClsModelFake2()]

        # Symbol initialization
        self.X = lambda i: int(gtsam.Symbol('x', i))
        self.XO = lambda j: int(gtsam.Symbol('o', j))

    def add_measurement(self, measurement, new_da_realization, graph, initial, da_realization, class_realization):
        """
        :type graph: gtsam.NonlinearFactorGraph
        :type initial: gtsam.Values
        """

        # Adding a factor per each measurement with corresponding data association realization
        measurement_number = 0
        #path_length = len(da_realization) - 1
        path_length = len(da_realization)

        for obj in new_da_realization:

            obj = int(obj)

            for obj_index in class_realization:
                if obj == obj_index[0]:
                    obj_class = obj_index[1]

            graph.add(libNewFactorFake_p1_3.ClsModelFactor3(self.X(path_length), self.XO(obj),
                                                           self.ClsModelCore[obj_class - 1],
                                                           list(measurement[measurement_number])))

            measurement_number += 1

            # Initialization of object of the realization
            # TODO: CHANGE TO SOMETHING MORE ACCURATE / LESS BASIC
            if initial.exists(self.XO(obj)) is False:
                initial.insert(self.XO(obj), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))
    def log_pdf_value(self, measurement, cls, point_x, point_xo):
        """

        :type point_x: gtsam.Pose3
        :type point_xo: gtsam.Pose3
        :type measurement: gtsam.Pose3
        """

        # Create an auxilary factor and compute his log-pdf value (without the normalizing constant)
        aux_factor = libNewFactorFake_p1_3.ClsModelFactor3(1, 2, self.ClsModelCore[cls - 1], list(measurement))
        poses = gtsam.Values()
        poses.insert(1, point_x)
        poses.insert(2, point_xo)

        return aux_factor.error(poses)

    # Runs the network for single relative pose input, also works if ML gamma is needed
    def model_output(self, pose_x, pose_xo, class_query):

        rel_pose = pose_xo.between(pose_x)
        rel_position = [rel_pose.x(), rel_pose.y(), rel_pose.z()]

        net_prediction = self.ClsModelCore[class_query-1].netOutputPy(rel_position)

        expectation = np.array(net_prediction[0:2])
        r_mat = np.array([[net_prediction[2], net_prediction[3]], [0, net_prediction[4]]])
        covariance = np.linalg.inv(np.matmul(np.transpose(r_mat), r_mat))

        return expectation, covariance

    # Samples gamma from model
    def model_sample(self, pose_x, pose_xo, class_query, ML_sample=False):

        [expectation, covariance] = self.model_output(pose_x, pose_xo, class_query)
        sem_gen = -1

        if ML_sample is False:
            while sem_gen < 0 or sem_gen > 1:
                sem_gen = np.random.normal(expectation[0], covariance[0, 0])

            output = np.array([sem_gen, 1 - sem_gen])

            return output

        else:
            return expectation







