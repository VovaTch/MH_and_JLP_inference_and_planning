from __future__ import print_function
import gtsam
from gtsam.symbol_shorthand import X, O
import numpy as np
# Fake classifier model import
import cls_un_model_4d

class ClsModel:

    def __init__(self, exp_add_1, exp_add_2, exp_add_3, exp_add_4, exp_add_5,
                 rinf_add_1, rinf_add_2, rinf_add_3, rinf_add_4, rinf_add_5):
        # Initialize the class
        cls_un_model_4d.initialize_python_objects(exp_add_1, exp_add_2, exp_add_3,
                                                                   exp_add_4, exp_add_5,
                                                                   rinf_add_1, rinf_add_2, rinf_add_3,
                                                                   rinf_add_4, rinf_add_5)

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

        exp_logit = np.exp(logit)

        probability_vector = np.divide(exp_logit, (1 + np.sum(exp_logit))).tolist()

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
            graph.add(cls_un_model_4d.ClsUModelFactor(self.X(path_length), self.XO(obj),
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
        aux_factor = cls_un_model_4d.ClsUModelFactor(1, 2, meas_logit, cls)

        poses = gtsam.Values()
        poses.insert(1, point_x)
        poses.insert(2, point_xo)

        return aux_factor.error(poses) #- meas_prod

    def log_pdf_value_new(self, measurement, cls, point_x, point_xo):

        logit = self.prob_vector_2_logit(measurement)
        logit_np = np.array(logit)

        [expectation, covariance] = self.model_output(point_x, point_xo, cls)

        mul_temp = np.matmul(expectation - logit_np, np.linalg.inv(covariance))
        exp_dif = expectation - logit_np
        mul_comp = np.dot(mul_temp, exp_dif.transpose())
        #mul_comp = mul_comp[0, 0]

        return mul_comp

    # Construct a covariance matrix from a flatten covariance
    @staticmethod
    def unflatten_rinf(straight_cov, matrix_len=4):

        matrix = np.zeros((matrix_len, matrix_len))
        counter = 0
        for idx_1 in range(matrix_len):
            for idx_2 in range(idx_1, matrix_len):
                matrix[idx_1, idx_2] = straight_cov[counter]
                counter += 1

        return matrix

    # Construct a covariance matrix from a flatten covariance
    @staticmethod
    def unflatten_cov(straight_cov, matrix_len=4):

        matrix = np.zeros((matrix_len, matrix_len))
        counter = 0
        for idx_1 in range(matrix_len):
            for idx_2 in range(idx_1, matrix_len):
                matrix[idx_1, idx_2] = straight_cov[counter]
                matrix[idx_2, idx_1] = straight_cov[counter]
                counter += 1

        return matrix

    # Runs the network for single relative pose input, also works if ML gamma is needed
    def model_output(self, pose_x, pose_xo, class_query):

        rel_pose = pose_xo.between(pose_x)
        rel_position = [rel_pose.x(), rel_pose.y(), 0.0] #rel_pose.z()]

        #print('class querry: ' + str(class_query))
        #print('rel pos: ' + str(rel_position))

        if class_query == 1:
            net_prediction = cls_un_model_4d.AuxOutputs.net_output_1(rel_position)
        elif class_query == 2:
            net_prediction = cls_un_model_4d.AuxOutputs.net_output_2(rel_position)
        elif class_query == 3:
            net_prediction = cls_un_model_4d.AuxOutputs.net_output_3(rel_position)
        elif class_query == 4:
            net_prediction = cls_un_model_4d.AuxOutputs.net_output_4(rel_position)
        elif class_query == 5:
            net_prediction = cls_un_model_4d.AuxOutputs.net_output_5(rel_position)

        expectation = np.array(net_prediction[0: 4])
        rinf = self.unflatten_rinf(net_prediction[4: 14])
        covariance = np.linalg.inv(np.matmul(rinf.transpose(), rinf))

        #print('expectation: ' + str(expectation))
        #print('covariance: ' + str(covariance))

        return expectation, covariance

    # Samples gamma from model
    def model_sample(self, pose_x, pose_xo, class_query, ML_sample=False):

        [expectation, covariance] = self.model_output(pose_x, pose_xo, class_query)

        sem_gen = np.random.multivariate_normal(expectation, covariance)

        if not isinstance(sem_gen, list):
            sem_gen = [sem_gen]

        if ML_sample is False:

            return self.logit_2_probability_vector(sem_gen)

        else:
            return self.logit_2_probability_vector(expectation)




