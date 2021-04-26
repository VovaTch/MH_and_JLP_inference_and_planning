from __future__ import print_function
import gtsam
import numpy as np
# Fake classifier model import
import libNewFactor_LG, libNewFactor_LG_rand2, libNewFactor_LG_rand3

class ClsModel:

    def __init__(self, addresses):
        # Initializing the classifier models, one per class
        libNewFactor_LG.initialize_python_objects(addresses[0], addresses[1])
        libNewFactor_LG_rand2.initialize_python_objects(addresses[2], addresses[3])
        libNewFactor_LG_rand3.initialize_python_objects(addresses[4], addresses[5])
        self.ClsModelCore = [libNewFactor_LG, libNewFactor_LG_rand2, libNewFactor_LG_rand3]

        # Symbol initialization
        self.X = lambda i: int(gtsam.Symbol('x', i))
        self.XO = lambda j: int(gtsam.Symbol('o', j))

    # Convert probability vector to logit
    @staticmethod
    def prob_vector_2_logit(probability_vector):

        logit_measurement = list()

        for cls in range(len(probability_vector) - 1):
            logit_measurement.append(np.log(probability_vector[cls] / probability_vector[-1]))
            if logit_measurement[-1] > 300:
                logit_measurement[-1] = 300
            elif logit_measurement[-1] < -300:
                logit_measurement[-1] = -300

        return logit_measurement

    # Convert logit to probability vector
    @staticmethod
    def logit_2_probability_vector(logit):


        probability_vector = list()
        exp_logit = np.exp(logit[0])
        exp_logit = np.ndarray.tolist(exp_logit)

        for cls in range(len(exp_logit)):
            probability_vector.append(exp_logit[cls] / (1 + np.sum(exp_logit)))

        probability_vector.append(1 / (1 + np.sum(exp_logit)))

        return probability_vector

    def add_measurement(self, measurement, new_da_realization, graph, initial, da_realization, class_realization):
        """

        :type graph: gtsam.NonlinearFactorGraph
        :type initial: gtsam.Values
        """

        # Adding a factor per each measurement with corresponding data association realization
        measurement_number = 0
        path_length = len(da_realization) - 1

        for obj in new_da_realization:

            obj = int(obj)

            for obj_index in class_realization:
                if obj == obj_index[0]:
                    obj_class = obj_index[1]
                    obj_class = int(obj_class)

            logit_measurement = self.prob_vector_2_logit(measurement[measurement_number])
            graph.add(self.ClsModelCore[obj_class - 1].ClsModelFactor3(self.X(path_length), self.XO(obj),
                                                                       list(logit_measurement)))
        #
        # for realization in new_da_realization:
        #     realization = int(realization)
        #     graph.add(self.ClsModelCore[class_realization[new_da_realization[measurement_number]-1]-1].
        #               ClsModelFactor3(self.X(path_length), self.XO(realization), measurement[measurement_number]))
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
        aux_factor = self.ClsModelCore[cls - 1].ClsModelFactor3(1, 2, measurement)
        poses = gtsam.Values()
        poses.insert(1, point_x)
        poses.insert(2, point_xo)
        # return self.ClsModelCore[cls-1].error_out(point_xo.between(point_x), measurement)

        return aux_factor.error(poses)

    def log_pdf_value_new(self, measurement, cls, point_x, point_xo):

        logit = self.prob_vector_2_logit(measurement)
        logit_np = np.array(logit)

        [expectation, covariance] = self.model_output(point_x, point_xo, cls)

        mul_temp = np.matmul(expectation - logit_np, np.linalg.inv(covariance))
        mul_comp = np.dot(mul_temp, expectation - logit_np)

        return mul_comp

    # Runs the network for single relative pose input, also works if ML gamma is needed
    def model_output(self, pose_x, pose_xo, class_query):

        rel_pose = pose_xo.between(pose_x)
        rel_position = [rel_pose.x(), rel_pose.y(), rel_pose.z()]

        net_prediction = self.ClsModelCore[class_query-1].netOutputPy(rel_position)

        expectation = np.array(net_prediction[0:2])
        r_mat = np.array([[net_prediction[2],net_prediction[3]], [0, net_prediction[4]]])
        covariance = np.linalg.inv(np.matmul(np.transpose(r_mat), r_mat))

        return expectation, covariance

    # Samples gamma from model
    def model_sample(self, pose_x, pose_xo, class_query, ML_sample=False):

        [expectation, covariance] = self.model_output(pose_x, pose_xo, class_query)
        #sem_gen = np.random.normal(expectation[0], covariance[0, 0])
        sem_gen = np.random.multivariate_normal(expectation, covariance)

        if not isinstance(sem_gen, list):
            sem_gen = [sem_gen]

        if ML_sample is False:

            return self.logit_2_probability_vector(sem_gen)

        else:
            return self.logit_2_probability_vector(expectation)
