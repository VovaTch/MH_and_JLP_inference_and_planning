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

    # Extract Dirichlet hyperparameters of an object classification probabilities
    def object_lambda_prob_dirichlet(self, selected_object, anti_zero_epsilon = 1e-100):

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
        return self.dirichlet_MLE(lambda_list)

    # Extract Dirichlet hyperparameters of class realization probabilities
    def realization_lambda_prob_dirichlet(self, anti_zero_epsilon = 1e-100):

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

        return self.dirichlet_MLE(lambda_list)

    # MLE of dirichlet
    def dirichlet_MLE(self, lambda_list, num_iter = 100):

        log_lambda = np.log(lambda_list)
        log_lambda_sum = np.sum(log_lambda, 0) / self.number_of_beliefs

        alpha_start = np.ones(np.size(lambda_list[0, :]))

        for idx in range(num_iter):
            digamma_sum_alpha = digamma(sum(alpha_start))
            for cls in range(np.size(lambda_list[0, :])):
                alpha_start[cls] = self.inv_digamma(digamma_sum_alpha + log_lambda_sum[cls])

        return alpha_start

    # Get entropy reward
    def entropy(self, select_object, anti_zero_epsilon=1e-100):

        alphas = self.object_lambda_prob_dirichlet(select_object, anti_zero_epsilon=anti_zero_epsilon)
        number_of_classes = len(alphas)

        entropy = - sp.special.gammaln(np.sum(alphas)) + \
                  (np.sum(alphas) - number_of_classes) * sp.special.psi(np.sum(alphas))
        for idx in range(number_of_classes):
            entropy += sp.special.gammaln(alphas[idx]) - (alphas[idx] - 1) * sp.special.psi(alphas[idx])

        return entropy

    # Get entropy reward for entire realizations
    def entropy_realizations(self, anti_zero_epsilon=1e-100):

        for belief in self.belief_list:
            object_list = belief.obj_realization
            break

        # If no objects observed yet return 0
        if 'object_list' not in locals():
            return 0

        alphas = self.realization_lambda_prob_dirichlet(anti_zero_epsilon=anti_zero_epsilon)
        number_of_realizations  = len(alphas)

        entropy = - sp.special.gammaln(np.sum(alphas)) + \
                  (np.sum(alphas) - number_of_realizations) * sp.special.psi(np.sum(alphas))

        for idx in range(number_of_realizations):
            entropy += sp.special.gammaln(alphas[idx]) - (alphas[idx] - 1) * sp.special.psi(alphas[idx])

        return entropy


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
                               plot_individual_points=False):

        for belief in self.belief_list:
            object_list = belief.obj_realization
            break

        # If no objects observed yet return 0
        if 'object_list' not in locals():
            return 0

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

        # Plot the data
        for cls_idx in range(number_of_bars):
            if cls_idx == 0:
                bars = plotterdaac2d.plot_weight_bars_multi(ax, cls_weights_list[cls_idx + 1], labels=x_labels,
                                                            log_weight_flag=False,
                                                            plot_individual_points=plot_individual_points)
            else:
                bars = plotterdaac2d.plot_weight_bars_multi(ax, cls_weights_list[cls_idx + 1], labels=x_labels,
                                                            log_weight_flag=False, bottom=cls_weights_list[cls_idx],
                                                            plot_individual_points=plot_individual_points)
        x = np.arange(belief.number_of_objects)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        # Rotate labes
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

        if GT_classes is not None:
            idx = 0
            for rect in bars:
                plt.text(rect.get_x() + rect.get_width() / 2.7, height[idx], 'X')

                idx += 1

        if show_plt is True:
            plt.tight_layout()
            if plt_save_name is not None:
                plt.savefig(plt_save_name + '.eps', format='eps')
            plt.show(block=False)
            if pause_time is not None:
                plt.pause(pause_time)
                plt.close()

    # Clone function necessary for the planning
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

        alpha = self.object_lambda_prob_dirichlet(selected_object)
        if write_lambdas is True:
            print('Dirichlet hyperparameters: ' + str(alpha))

        B_alpha = sp.special.gamma(alpha[0]) * sp.special.gamma(alpha[1]) / sp.special.gamma(alpha[0] + alpha[1])

        x_array = np.arange(1e-4, 1e-3, 1e-4)
        x_array = np.concatenate((x_array, np.arange(1e-3, 1-1e-3, 1e-3)), axis=0)
        x_array = np.concatenate((x_array, np.arange(1-1e-3, 1-1e-4, 1e-4)), axis=0)
        y_array = np.zeros(len(x_array))

        for idx in range(len(x_array)):
            y_array[idx] = x_array[idx] ** (alpha[0] - 1) * (1 - x_array[idx]) ** (alpha[1] - 1) / B_alpha

        plt.grid(True)
        plt.xlabel('Probability of class 1')
        plt.ylabel('Dirichlet Probability')
        plt.ylim([0, 5])
        plt.plot(x_array, y_array, alpha=alpha_g, color='black', linewidth=2)
        if show_plot is True:
            plt.show()







