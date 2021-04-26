from __future__ import print_function
import numpy as np
import scipy as sp
import itertools
import gtsam
import math
import copy
import plotterdaac2d
import colorsys
import matplotlib.pyplot as plt
from gaussianb_isam2 import GaussianBelief
#from gaussianb import GaussianBelief
import matplotlib
import dirichlet
from scipy.special import digamma, polygamma
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



class LambdaBelief:

    # Creating the collection of beliefs. Each belief can have it's own models, they are defined
    def __init__(self, belief_list, cls_enable=True):

        self.belief_list = belief_list
        self.number_of_beliefs = len(belief_list)
        self.cls_enable = cls_enable

    # Forwarding an action
    def action_step(self, action, action_noise):

        for belief_idx in range(self.number_of_beliefs):
            self.belief_list[belief_idx].action_step(action, action_noise)

    # Adding measurements; geometric measurements have the same format as Hybrid Belief.
    # Semantic measurements will be in the format of measurement_list[ind_gamma][object]
    def add_measurements(self, geo_measurements, sem_measurements, da_current_step, number_of_samples=10,
                         new_input_object_prior=None, new_input_object_covariance=None):

        for belief_idx in range(self.number_of_beliefs):
            self.belief_list[belief_idx].add_measurements(geo_measurements, sem_measurements[belief_idx], da_current_step,
                                                          number_of_samples, new_input_object_prior,
                                                          new_input_object_covariance)

    def cls_probability_total(self):

        # Gets a cloud of Lambdas
        lambda_list = np.zeros([self.number_of_beliefs,  self.belief_list[0].number_of_classes **
                                self.belief_list[0].number_of_objects])
        for belief_idx in range(self.number_of_beliefs):
            # Extract probability vectors for all realizations
            cls_weights = self.belief_list[belief_idx].compute_cls_weights()
            idx = 0
            for realization in cls_weights:
                lambda_list[belief_idx][idx] = cls_weights[realization]
                idx += 1

        return lambda_list

    # Create a matrix of class probability vectors for an object
    def cls_probability_object(self, selected_object):

        # Gets a cloud of lambdas
        lambda_list = np.zeros([self.number_of_beliefs, self.belief_list[0].number_of_classes])
        for belief_idx in range(self.number_of_beliefs):
            # Extract probability vectors for all realizations
            cls_weights = self.belief_list[belief_idx].compute_cls_weights()
            for realization in cls_weights:
                obj_class = self.belief_list[belief_idx].find_object_cls_in_realization(realization, selected_object)
                if obj_class is not None:
                    lambda_list[belief_idx][int(obj_class[1]) - 1] += cls_weights[realization]
                else:
                    return None

        return lambda_list

    # Extract Logistical Gaussian parameters of an object classification probabilities
    def object_lambda_prob_lg(self, selected_object, anti_zero_epsilon = 1e-100):

        # Gets a cloud of lambdas
        lambda_list = np.zeros([self.number_of_beliefs, self.belief_list[0].number_of_classes])
        for belief_idx in range(self.number_of_beliefs):
            # Extract probability vectors for all realizations
            cls_weights = self.belief_list[belief_idx].compute_cls_weights()
            for realization in cls_weights:
                obj_class = self.belief_list[belief_idx].find_object_cls_in_realization(realization, selected_object)
                if obj_class is not None:
                    lambda_list[belief_idx][int(obj_class[1]) - 1] += cls_weights[realization]
                else:
                    return np.ones(self.belief_list[belief_idx].number_of_classes)

        # To prevent the algorithm from crashing
        lambda_list = lambda_list + anti_zero_epsilon
        lambda_list = lambda_list / np.sum(lambda_list, axis=1)[:, np.newaxis]

        #return dirichlet.mle(lambda_list, maxiter=100, tol=1e3)
        return self.lg_MLE(lambda_list)

    # Extract Dirichlet hyperparameters of class realization probabilities
    def realization_lambda_prob_lg(self, anti_zero_epsilon = 1e-100):

        # Gets a cloud of Lambdas
        lambda_list = np.zeros([self.number_of_beliefs,  self.belief_list[0].number_of_classes **
                                self.belief_list[0].number_of_objects])
        for belief_idx in range(self.number_of_beliefs):
            # Extract probability vectors for all realizations
            cls_weights = self.belief_list[belief_idx].compute_cls_weights()
            idx = 0
            for realization in cls_weights:
                lambda_list[belief_idx][idx] = cls_weights[realization]
                idx += 1

        # To prevent the algorithm from crashing
        lambda_list = lambda_list + anti_zero_epsilon
        lambda_list = lambda_list / np.sum(lambda_list, axis=1)[:, np.newaxis]

        return self.lg_MLE(lambda_list)

    # MLE of logistical gaussian parameters
    def lg_MLE(self, lambda_list):

        dimensions = lambda_list.shape

        logistical_lambda = np.log(lambda_list[:, 0:-1]) - np.log(lambda_list[:, -1])[:, np.newaxis]
        expectation = np.sum(logistical_lambda, axis=0) / dimensions[0]

        distance_from_expectation = logistical_lambda - expectation
        covariance = np.matmul(np.transpose(distance_from_expectation), distance_from_expectation) / (dimensions[0] - 1)

        return expectation, covariance

    # Get entropy reward
    def entropy(self, select_object, anti_zero_epsilon=1e-10, entropy_lower_limit=None):

        expectation, covariance = self.object_lambda_prob_lg(select_object)

        lambda_list = self.cls_probability_object(select_object)
        lambda_list = lambda_list + anti_zero_epsilon
        lambda_list = lambda_list / np.sum(lambda_list, axis=1)[:, np.newaxis]

        lambda_prod_div = np.divide(1, np.prod(lambda_list, axis=1))

        logistical_lambda = np.log(lambda_list[:, 0:-1]) - np.log(lambda_list[:, -1])[:, np.newaxis]
        lg_dif = logistical_lambda - expectation

        sign , log_pdf_constant = np.linalg.slogdet(2 * np.pi * covariance)

        log_pdf = -0.5 * log_pdf_constant + np.log(lambda_prod_div)

        for idx in range(len(lambda_prod_div)):
            first_mul = np.matmul(lg_dif[idx, :], np.linalg.inv(covariance +
                                                                anti_zero_epsilon * np.eye(len(expectation))))
            second_mul = np.dot(first_mul, lg_dif[idx, :])
            log_pdf[idx] += -0.5 * second_mul

        entropy = - np.sum(log_pdf) / len(lambda_prod_div)
        if entropy_lower_limit is not None and entropy < entropy_lower_limit:
            entropy = entropy_lower_limit

        return entropy

    # Get entropy reward for entire realizations
    def entropy_realizations(self, anti_zero_epsilon=1e-10, entropy_lower_limit=None):

        for belief in self.belief_list:
            object_list = belief.obj_realization
            break

        # If no objects observed yet return 0
        if 'object_list' not in locals():
            return 0

        expectation, covariance = self.realization_lambda_prob_lg(anti_zero_epsilon=anti_zero_epsilon)

        lambda_list = self.cls_probability_total()
        lambda_list = lambda_list + anti_zero_epsilon
        lambda_list = lambda_list / np.sum(lambda_list, axis=1)[:, np.newaxis]

        lambda_prod_div = np.divide(1, np.prod(lambda_list, axis=1))

        logistical_lambda = np.log(lambda_list[:, 0:-1]) - np.log(lambda_list[:, -1])[:, np.newaxis]
        lg_dif = logistical_lambda - expectation

        sign, log_pdf_constant = np.linalg.slogdet(2 * np.pi * covariance +
                                                   anti_zero_epsilon * np.eye(covariance.shape[0]))

        log_pdf = -0.5 * log_pdf_constant + np.log(lambda_prod_div)

        for idx in range(len(lambda_prod_div)):
            first_mul = np.matmul(lg_dif[idx, :], np.linalg.inv(covariance +
                                                                anti_zero_epsilon * np.eye(covariance.shape[0])))
            second_mul = np.dot(first_mul, lg_dif[idx, :])
            log_pdf[idx] += -0.5 * second_mul

        entropy = - np.sum(log_pdf) / len(lambda_prod_div)
        if entropy_lower_limit is not None and entropy_lower_limit < entropy:
            entropy = entropy_lower_limit
        # entropy = 0.5 * np.linalg.slogdet(2 * np.pi * np.exp(1) * covariance
        #                                 + anti_zero_epsilon * np.eye(covariance.shape[0]))[1] - np.abs(expectation) #TODO: FIT MORE THAN 2 VARIABLES

        return entropy

    # Get entropy reward for expectation of lambda
    def entropy_lambda_expectation(self, selected_object, number_of_samples=100):

        # Gets a cloud of lambdas
        lambda_list = np.zeros([self.number_of_beliefs, self.belief_list[0].number_of_classes])
        for belief_idx in range(self.number_of_beliefs):
            # Extract probability vectors for all realizations
            cls_weights = self.belief_list[belief_idx].compute_cls_weights()
            for realization in cls_weights:
                obj_class = self.belief_list[belief_idx].find_object_cls_in_realization(realization, selected_object)
                if obj_class is not None:
                    lambda_list[belief_idx][int(obj_class[1]) - 1] += cls_weights[realization]
                else:
                    return np.ones(self.belief_list[belief_idx].number_of_classes)

        # To prevent the algorithm from crashing
        lambda_list = lambda_list + 1e-100
        lambda_list = lambda_list / np.sum(lambda_list, axis=1)[:, np.newaxis]

        # Compute expectation
        lambda_exp = np.sum(lambda_list, axis=0) / self.number_of_beliefs
        entropy = 0
        for cls_idx in range(len(lambda_exp)):
            entropy -= lambda_exp[cls_idx] * np.log(lambda_exp[cls_idx])

        return entropy

    # Get the covariance of Lambda of an object
    def lambda_covariance(self, selected_object, number_of_samples=100):

        # Gets a cloud of lambdas
        lambda_list = np.zeros([self.number_of_beliefs, self.belief_list[0].number_of_classes])
        for belief_idx in range(self.number_of_beliefs):
            # Extract probability vectors for all realizations
            cls_weights = self.belief_list[belief_idx].compute_cls_weights()
            for realization in cls_weights:
                obj_class = self.belief_list[belief_idx].find_object_cls_in_realization(realization, selected_object)
                if obj_class is not None:
                    lambda_list[belief_idx][int(obj_class[1]) - 1] += cls_weights[realization]
                else:
                    return np.ones(self.belief_list[belief_idx].number_of_classes)

        # To prevent the algorithm from crashing
        lambda_list = lambda_list + 1e-100
        lambda_list = lambda_list / np.sum(lambda_list, axis=1)[:, np.newaxis]

        # Compute expectation
        lambda_exp = np.sum(lambda_list, axis=0) / self.number_of_beliefs
        lambda_exp_tile = np.tile(lambda_exp, (self.number_of_beliefs, 1))
        lambda_cov = np.matmul(np.transpose(lambda_list - lambda_exp_tile), lambda_list - lambda_exp_tile) \
                     / self.number_of_beliefs

        #print(lambda_list)

        return lambda_cov

    # Get entropy as a sum for individual object entropy
    def entropy_lambda_expectation_total(self):

        entropy = 0
        for belief in self.belief_list:
            object_list = belief.obj_realization
            break

        for obj in object_list:
            entropy += self.entropy_lambda_expectation(obj)

        return entropy

    # MSDE for a single object
    def MSDE_obj(self, obj, GT_class):

        for belief in self.belief_list:
            number_of_classes = belief.number_of_classes
            break

        MSDE = 0

        # Gets a cloud of lambdas
        lambda_list = np.zeros([self.number_of_beliefs, self.belief_list[0].number_of_classes])
        for belief_idx in range(self.number_of_beliefs):
            # Extract probability vectors for all realizations
            cls_weights = self.belief_list[belief_idx].compute_cls_weights()
            for realization in cls_weights:
                obj_class = self.belief_list[belief_idx].find_object_cls_in_realization(realization, obj)
                if obj_class is not None:
                    lambda_list[belief_idx][int(obj_class[1]) - 1] += cls_weights[realization]
                else:
                    return np.ones(self.belief_list[belief_idx].number_of_classes)

        # To prevent the algorithm from crashing
        lambda_list = lambda_list + 1e-100
        lambda_list = lambda_list / np.sum(lambda_list, axis=1)[:, np.newaxis]

        # Compute expectation
        lambda_exp = np.sum(lambda_list, axis=0) / self.number_of_beliefs

        for cls_idx in range(number_of_classes):

            cls_true = 0
            if GT_class == cls_idx + 1:
                cls_true = 1

            MSDE += (lambda_exp[cls_idx] - cls_true) ** 2 / number_of_classes

        return MSDE

    # MSDE metric for classification accuracy
    def MSDE_expectation(self, GT_Realization):

        for belief in self.belief_list:
            object_list = belief.obj_realization
            number_of_classes = belief.number_of_classes
            break

        MSDE = 0
        for obj in GT_Realization:

            if obj in object_list:

                # Gets a cloud of lambdas
                lambda_list = np.zeros([self.number_of_beliefs, self.belief_list[0].number_of_classes])
                for belief_idx in range(self.number_of_beliefs):
                    # Extract probability vectors for all realizations
                    cls_weights = self.belief_list[belief_idx].compute_cls_weights()
                    for realization in cls_weights:
                        obj_class = self.belief_list[belief_idx].find_object_cls_in_realization(realization, obj)
                        if obj_class is not None:
                            lambda_list[belief_idx][int(obj_class[1]) - 1] += cls_weights[realization]
                        else:
                            return np.ones(self.belief_list[belief_idx].number_of_classes)

                # To prevent the algorithm from crashing
                lambda_list = lambda_list + 1e-100
                lambda_list = lambda_list / np.sum(lambda_list, axis=1)[:, np.newaxis]

                # Compute expectation
                lambda_exp = np.sum(lambda_list, axis=0) / self.number_of_beliefs

                for cls_idx in range(number_of_classes):

                    cls_true = 0
                    if GT_Realization[obj] == cls_idx + 1:
                        cls_true = 1

                    MSDE += (lambda_exp[cls_idx] - cls_true) ** 2 / number_of_classes / len(GT_Realization)

            else:

                for cls_idx in range(number_of_classes):

                    cls_true = 0
                    if GT_Realization[obj] == cls_idx + 1:
                        cls_true = 1

                    MSDE += (1 / number_of_classes - cls_true) ** 2 / number_of_classes / len(GT_Realization)


        return MSDE

    # Average entropy, helpful for planning
    def avg_entropy(self, anti_zero_epsilon=1e-100):

        for belief in self.belief_list:
            object_list = belief.obj_realization
            break

        avg_entropy = 0
        number_of_objects = len(object_list)
        for obj in object_list:
            avg_entropy += self.entropy(obj, anti_zero_epsilon=anti_zero_epsilon) / number_of_objects

        return avg_entropy

    # Inverse digamma
    @staticmethod
    def inv_digamma(y, eps=1e-8, max_iter=100):
        '''Numerical inverse to the digamma function by root finding'''

        if y >= -2.22:
            xold = np.exp(y) + 0.5
        else:
            xold = -1 / (y - digamma(1))

        for _ in range(max_iter):

            xnew = xold - (digamma(xold) - y) / polygamma(1, xold)

            if np.abs(xold - xnew) < eps:
                break

            xold = xnew

        return xnew

    # Create bar graph with ranges
    def lambda_bar_error_graph(self, fig_num = 0, show_plt=True, plt_save_name=None, pause_time=None, GT_classes=None,
                               plot_individual_points=False, ax=None, color='blue', width_offset=0, GT_bar=False):
        for belief in self.belief_list:
            object_list = belief.obj_realization
            break

        # If no objects observed yet return 0
        if 'object_list' not in locals():
            return 0

        if ax is None:
            fig = plt.figure(fig_num)

            ax = fig.gca()
            plt.gca()
            plt.rcParams.update({'font.size': 16})

        number_of_bars = self.belief_list[0].number_of_classes - 1
        number_of_objects = len(object_list)

        # Create plot-able data segments
        cls_weights_list = dict()
        for belief in self.belief_list:
            cls_weights = belief.compute_cls_weights()
            for cls_idx in range(number_of_bars):
                cls_weights_list[cls_idx + 1] = np.zeros([self.number_of_beliefs, belief.number_of_objects])
            break

        for belief_idx in range(len(self.belief_list)):
            x_labels = list()
            height = list()
            cls_weights_single = self.belief_list[belief_idx].compute_cls_weights()
            for realization in cls_weights_single:
                for idx in range(belief.number_of_objects):
                    if realization[idx][1] != belief.number_of_classes:
                        cls_weights_list[realization[idx][1]][belief_idx][idx] += cls_weights_single[realization]
                    x_labels.append('Object ' + str(realization[idx][0]))

                    if GT_classes is not None:
                        if GT_classes[realization[idx][0]] == 1:
                            height.append(1)
                        if GT_classes[realization[idx][0]] == 2:
                            height.append(0)

        # If GT_bar is true, show bar relative to the GT class.
        if GT_bar is True:
            for realization in cls_weights_single:
                for cls_idx in range(number_of_bars):
                    for idx in range(belief.number_of_objects):
                        if cls_idx + 1 != GT_classes[realization[idx][0]]:
                            cls_weights_list[cls_idx + 1][:, idx] = \
                                1 - cls_weights_list[cls_idx + 1][:, idx]
                break

        # Plot the data
        for cls_idx in range(number_of_bars):
            if cls_idx == 0:
                bars = plotterdaac2d.plot_weight_bars_multi(ax, cls_weights_list[cls_idx + 1], labels=x_labels,
                                                            log_weight_flag=False,
                                                            plot_individual_points=plot_individual_points, color=color,
                                                            width_offset = width_offset)
            else:
                bars = plotterdaac2d.plot_weight_bars_multi(ax, cls_weights_list[cls_idx + 1], labels=x_labels,
                                                            log_weight_flag=False, bottom=cls_weights_list[cls_idx],
                                                            plot_individual_points=plot_individual_points, color=color,
                                                            width_offset = width_offset)
        x = np.arange(belief.number_of_objects)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        # Rotate labes
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

        if GT_classes is not None and GT_bar is False:
            idx = 0
            for rect in bars:
                #plt.text(rect.get_x() + rect.get_width() / 2.7, height[idx], 'X')
                plt.text(x[idx] - 0.13, height[idx], 'X')

                idx += 1

        if show_plt is True:
            plt.tight_layout()
            if plt_save_name is not None:
                plt.savefig(plt_save_name + '.eps', format='eps')
            plt.show(block=False)
            if pause_time is not None:
                plt.pause(pause_time)
                plt.close()

        return ax

    # # log PDF value of the logistical gaussian
    # @staticmethod
    # def log_pdf_lg(probability_vectors):
    #
    #     lg_probability_vectors = np.log(probability_vectors[:, 0:-1]) - \
    #                              np.log(probability_vectors[:, -1])[:, np.newaxis]
    #
    #
    #     pass

    # Clone function necessary for the planning

    # Create bar graph with ranges, gt if more than one class
    def lambda_bar_error_graph_multi(self, fig_num=0, show_plt=True, plt_save_name=None, pause_time=None, GT_classes=None,
                               plot_individual_points=False, ax=None, color='blue', width_offset=0, GT_bar=False):
        for belief in self.belief_list:
            object_list = belief.obj_realization
            break

        # If no objects observed yet return 0
        if 'object_list' not in locals():
            return 0

        if ax is None:
            fig = plt.figure(fig_num)

            ax = fig.gca()
            plt.gca()
            plt.rcParams.update({'font.size': 16})

        number_of_bars = self.belief_list[0].number_of_classes - 1
        number_of_objects = len(object_list)

        # Create plot-able data segments
        cls_weights_list = dict()
        for belief in self.belief_list:
            cls_weights = belief.compute_cls_weights()
            for cls_idx in range(number_of_bars):
                cls_weights_list[cls_idx + 1] = np.zeros([self.number_of_beliefs, belief.number_of_objects])
            break

        for belief_idx in range(len(self.belief_list)):
            x_labels = list()
            height = list()
            cls_weights_single = self.belief_list[belief_idx].compute_cls_weights()
            for realization in cls_weights_single:
                for idx in range(belief.number_of_objects):
                    if realization[idx][1] != belief.number_of_classes:
                        cls_weights_list[realization[idx][1]][belief_idx][idx] += cls_weights_single[realization]
                    x_labels.append('Object ' + str(realization[idx][0]))

                    if GT_classes is not None:
                        if GT_classes[realization[idx][0]] == 1:
                            height.append(1)
                        if GT_classes[realization[idx][0]] == 2:
                            height.append(0)

        cls_weight_list_gt = np.zeros([self.number_of_beliefs, belief.number_of_objects])

        print(x_labels)

        # If GT_bar is true, show bar relative to the GT class.
        if GT_bar is True:
            for realization in cls_weights_single:
                for cls_idx in range(number_of_bars):
                    for idx in range(belief.number_of_objects):
                        if cls_idx + 1 == GT_classes[realization[idx][0]]:
                            cls_weight_list_gt[:, idx] = \
                                cls_weights_list[cls_idx + 1][:, idx]

                break


        cls_weights_dict = dict()
        for realization in cls_weights_single:
            for idx in range(belief.number_of_objects):
                cls_weights_dict[realization[idx][0]] = cls_weight_list_gt[:, idx]
            break

        cls_weight_list_gt_full = np.zeros([self.number_of_beliefs, len(GT_classes)])
        for idx, key_gt in enumerate(GT_classes):
            if key_gt in cls_weights_dict:
                cls_weight_list_gt_full[:, idx] = cls_weights_dict[key_gt]


        # Plot the data
        bars = plotterdaac2d.plot_weight_bars_multi(ax, cls_weight_list_gt_full, labels=None,
                                                    log_weight_flag=False,
                                                    plot_individual_points=plot_individual_points,
                                                    color=color,
                                                    width_offset=width_offset)

        # Writes X on bars where there are no measurements
        for obj_idx, obj in enumerate(GT_classes):
            if obj not in object_list:
                plt.text(obj_idx + width_offset - 0.05, 0.5, 'X', color=color)


        # bars = plotterdaac2d.plot_weight_bars_multi_dict(ax, cls_weights_dict, labels=x_labels,
        #                                             log_weight_flag=False,
        #                                             plot_individual_points=plot_individual_points,
        #                                             color=color,
        #                                             width_offset=width_offset)

        x = np.arange(len(GT_classes))
        ax.set_xticks(x)
        # print(list(ax.get_xticks()))

        labels = list()
        for key in GT_classes:
            labels.append('Object: ' + str(int(key)))

        ax.set_xticklabels(labels)
        # print(list(ax.get_xticklabels()))


        # Rotate labes
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

        if GT_classes is not None and GT_bar is False:
            idx = 0
            for rect in bars:
                # plt.text(rect.get_x() + rect.get_width() / 2.7, height[idx], 'X')
                plt.text(x[idx] - 0.13, height[idx], 'X')

                idx += 1

        if show_plt is True:
            plt.tight_layout()
            if plt_save_name is not None:
                plt.savefig(plt_save_name + '.eps', format='eps')
            plt.show(block=False)
            if pause_time is not None:
                plt.pause(pause_time)
                plt.close()

        return ax

    def clone(self):

        new_object = copy.copy(self)
        new_object.belief_list = [x.clone() for x in self.belief_list]
        for belief_idx in range(len(self.belief_list)):
            new_object.belief_list[belief_idx].belief_matrix = dict()
            for g_idx in self.belief_list[belief_idx].belief_matrix:
                new_object.belief_list[belief_idx].belief_matrix[g_idx] = \
                    self.belief_list[belief_idx].belief_matrix[g_idx].clone()
        return new_object

    # Extract the ML robot pose
    def ML_latest_poses(self):
        pass

    # 2 class simplex figure
    def simplex_2class(self, selected_object, write_lambdas=True, show_plot=True, alpha_g=1):

        fig = plt.figure(0)
        ax = fig.gca()

        expectation, covariance = self.object_lambda_prob_lg(selected_object)


        if write_lambdas is True:
            print('Mu: ' + str(expectation) + '; Covariance: ' + str(covariance))

        x_array = np.arange(1e-5, 1e-4, 1e-5)
        x_array = np.concatenate((x_array, np.arange(1e-4, 1-1e-4, 1e-3)), axis=0)
        x_array = np.concatenate((x_array, np.arange(1-1e-4, 1-1e-5, 1e-5)), axis=0)
        y_array = np.zeros(len(x_array))

        for idx in range(len(x_array)):
            y_array[idx] = 1 / np.sqrt(2 * np.pi * np.linalg.det(covariance + 1e-10)) * \
                           (1 / (x_array[idx] - x_array[idx] ** 2)) * \
                           np.exp(-0.5 * (np.log(x_array[idx] / (1 - x_array[idx])) - expectation) ** 2 /
                                  (covariance + 1e-10))

        plt.grid(True)
        plt.xlabel(fr'$\gamma^{selected_object}$')
        plt.ylabel(fr'$P(\gamma^{selected_object})$')
        plt.ylim([0, 5])
        plt.plot(x_array, y_array, alpha=alpha_g, color='black', linewidth=2)
        if show_plot is True:
            plt.show()

    # Print LG parameters
    def print_lambda_lg_params(self):

        for belief in self.belief_list:
            object_list = belief.obj_realization
            break

        for obj in object_list:
            exp, cov = self.object_lambda_prob_lg(obj)
            print('-------Object Index (' + str(obj) + ')--------------')
            print('Expectation: ' + str(exp))
            print('Covariance: ' + str(cov) + '\n')

    # Compute camera and object distance metric
    def distance_metric(self, GT_poses_robot, GT_poses_object):

        obj_distance = 0
        robot_distance = 0
        for idx in range(self.number_of_beliefs):
            obj_distance += self.belief_list[idx].obj_distance_all_weights(GT_poses_robot)
            robot_distance += self.belief_list[idx].distance_all_weights(GT_poses_object)

        return robot_distance, obj_distance
