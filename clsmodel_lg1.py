from __future__ import print_function
import gtsam
from gtsam.symbol_shorthand import X, O
import numpy as np
# Fake classifier model import
import cls_un_model_fake_1d

class ClsModel:

    def __init__(self):
        # Initialize the class
        cls_un_model_fake_1d.initialize_python_objects()

        # Symbol initialization
        self.X = lambda i: X(i)
        self.XO = lambda j: O(j)

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
        exp_logit = np.exp(logit)

        probability_vector.append(float(exp_logit / (1 + np.sum(exp_logit))))

        probability_vector.append(1 / (1 + np.sum(exp_logit)))

        return probability_vector

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

            logit_measurement = self.prob_vector_2_logit(measurement[measurement_number])
            graph.add(cls_un_model_fake_1d.ClsUModelFactor(self.X(path_length), self.XO(obj),
                                                           logit_measurement, obj_class))

            measurement_number += 1

            # Initialization of object of the realization
            # TODO: CHANGE TO SOMETHING MORE ACCURATE / LESS BASIC
            if initial.exists(self.XO(obj)) is False:
                initial.insert(self.XO(obj), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))
    def log_pdf_value(self, measurement, cls, point_x, point_xo, epsilon = 1e-10):
        """

        :type point_x: gtsam.Pose3
        :type point_xo: gtsam.Pose3
        """

        # Create an auxilary factor and compute his log-pdf value (without the normalizing constant)
        measurement = np.array(measurement)
        meas_logit = self.prob_vector_2_logit(measurement)
        meas_prod = np.log(np.prod(measurement + epsilon))
        aux_factor = cls_un_model_fake_1d.ClsUModelFactor(1, 2, meas_logit, cls)

        poses = gtsam.Values()
        poses.insert(1, point_x)
        poses.insert(2, point_xo)

        return aux_factor.error(poses) #- meas_prod

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

        if class_query == 1:
            net_prediction = cls_un_model_fake_1d.AuxOutputs.net_output_1(rel_position)
        else:
            net_prediction = cls_un_model_fake_1d.AuxOutputs.net_output_2(rel_position)

        expectation = net_prediction[0]
        rinf = np.array([[net_prediction[1]]])
        covariance = np.linalg.inv(np.matmul(rinf.transpose(), rinf))

        return expectation, covariance

    # Samples gamma from model
    def model_sample(self, pose_x, pose_xo, class_query, ML_sample=False):

        [expectation, covariance] = self.model_output(pose_x, pose_xo, class_query)
        sem_gen = np.random.normal(expectation, covariance[0, 0])

        if not isinstance(sem_gen, list):
            sem_gen = [sem_gen]

        if ML_sample is False:

            return self.logit_2_probability_vector(sem_gen)

        else:
            return self.logit_2_probability_vector(expectation)




