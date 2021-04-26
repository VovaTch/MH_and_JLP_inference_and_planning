from __future__ import print_function
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, O, L
import joint_lambda_pose_factor_fake_2d

# JLP factor use in python
class JLPModel:

    def __init__(self):

        # Model initialization
        joint_lambda_pose_factor_fake_2d.initialize_python_objects()

        # Symbol initialization
        self.X = lambda i: X(i)
        self.XO = lambda j: O(j)
        self.Lam = lambda k: L(k)

    # Lambda numbering is a000000b where a is the object and b is the number of time steps
    def add_measurements(self, measurement_exp, measurement_cov, new_da_realization, graph, initial, da_realization):
        """

        :type graph: gtsam.NonlinearFactorGraph
        :type initial: gtsam.Values
        """

        # Adding a factor per each measurement with corresponding data association realization
        measurement_number = 0
        path_length = len(da_realization)
        for realization in new_da_realization:
            realization = int(realization)
            lam_realization = int(realization * 1e6 + path_length)

            # Flatten the covariance matrix
            deploy_matrix = self.flatten_cov(measurement_cov[measurement_number])

            # Find the previous realization
            lam_realization_prev = self.find_prev_lambda(initial, path_length, realization)
            lam_realization_prev = int(lam_realization_prev)

            graph.add(joint_lambda_pose_factor_fake_2d.JLPFactor(self.X(path_length), self.XO(realization),
                                                              self.Lam(lam_realization_prev), self.Lam(lam_realization),
                                                              measurement_exp[measurement_number], deploy_matrix))
            measurement_number += 1

            # Initialization of object of the realization
            if initial.exists(self.Lam(lam_realization_prev)) is False:
                initial.insert(self.Lam(lam_realization_prev), np.array([0., 0.]))

            if initial.exists(self.Lam(lam_realization)) is False:
                initial.insert(self.Lam(lam_realization), np.array([0., 0.]))

    # Find the previous step
    def find_prev_lambda(self, initial, path_length, object):

        for idx in range(path_length - 1, -1, -1):
            if initial.exists(self.Lam(int(object * 1e6 + idx))):
                return object * 1e6 + idx

    # Flatten a covariance matrix:
    @staticmethod
    def flatten_cov(covariance_matrix, matrix_len=2):

        deploy_matrix = list()
        for idx_1 in range(matrix_len):
            for idx_2 in range(idx_1 + 1):
                deploy_matrix.append(covariance_matrix[idx_1, idx_2])

        return deploy_matrix


    # Find the log pdf value
    def log_pdf_value(self, measurement_exp, measurement_cov, point_x, point_xo, lambda_prev, lambda_cur):
        """

        :type point_x: gtsam.Pose3
        :type point_xo: gtsam.Pose3
        :type lambda_prev: gtsam.Point2
        :type lambda_cur: gtsam.Point2
        """
        aux_factor = joint_lambda_pose_factor_fake_2d.JLPFactor(1, 2, 3, 4, measurement_exp,
                                                             self.flatten_cov(measurement_cov))
        value = gtsam.Values()
        value.insert(1, point_x)
        value.insert(2, point_xo)
        value.insert(3, lambda_prev)
        value.insert(4, lambda_cur)

        return aux_factor.error(value)

    # Extract expectation and covariance of classifier uncertainty model
    def sample_lambda_model(self, chosen_cls, relative_position):

        if chosen_cls == 1:
            output_list = joint_lambda_pose_factor_fake_2d.AuxOutputs.net_output_1(relative_position)
        elif chosen_cls == 2:
            output_list = joint_lambda_pose_factor_fake_2d.AuxOutputs.net_output_2(relative_position)
        elif chosen_cls == 3:
            output_list = joint_lambda_pose_factor_fake_2d.AuxOutputs.net_output_3(relative_position)

        exp = [output_list[0], output_list[1]]
        rinf = np.array([[output_list[2], output_list[3]], [0., output_list[4]]])
        cov = np.linalg.inv(np.matmul(rinf.transpose(), rinf))

        return exp, cov



