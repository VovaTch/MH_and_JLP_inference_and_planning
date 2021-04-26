from __future__ import print_function
import numpy as np
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
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



class HybridBelief:

    def __init__(self, number_of_classes, geo_model, da_model, cls_model, class_probability_prior,
                 prior_mean, prior_noise, pruning_threshold=5, cls_enable=True, norm_epsilon=0.00001, ML_update=False):

        # Creating a flag object realization matrix
        self.obj_realization = dict()

        # Setting up geometric, DA, and classifier models if needed
        self.geo_model = geo_model
        self.da_model = da_model
        self.cls_model = cls_model

        # Set weight pruning threshold, da flag and classifier flag
        self.pruning_threshold = pruning_threshold
        self.cls_enable = cls_enable
        self.norm_epsilon = norm_epsilon
        self.ML_update = ML_update

        # Creating a dictionary of all prior gaussian beliefs, filtering out those with 0 probability (if there are)
        self.belief_matrix = dict()
        self.belief_matrix[((0,),)] = GaussianBelief(class_probability_prior, geo_model, da_model, cls_model, prior_mean,
                                               prior_noise, cls_enable=self.cls_enable)

        # Setting variables for number of classes and objects
        self.number_of_classes = number_of_classes
        self.number_of_objects = 0

        # Empty belief flag
        self.empty_belief_flag = False

    # Forwarding the belief one step
    def action_step(self, action, action_noise):

        for prior_gaussian in self.belief_matrix:
            self.belief_matrix[prior_gaussian].action_step(action, action_noise)

    # Inserting new measurements
    def add_measurements(self, geo_measurements, sem_measurements, da_current_step, number_of_samples=10,
                         new_input_object_prior=None, new_input_object_covariance=None):

        # Check which objects are new
        number_of_new_objects = 0
        for obj in da_current_step:
            if obj not in self.obj_realization:
                self.obj_realization[obj] = 'new'
                self.number_of_objects += 1
                number_of_new_objects += 1
            else:
                self.obj_realization[obj] = 'old'

        for obj in self.obj_realization:
            if obj not in da_current_step:
                self.obj_realization[obj] = 'old'

        number_of_observations = len(geo_measurements)
        # number_of_prior_gaussians = len(self.belief_matrix)

        # Create class realizations
        cls_realizations = np.array(list(itertools.product(range(self.number_of_classes),
                                                                repeat=number_of_new_objects)))
        cls_realizations += 1


        if self.cls_enable is False:
            cls_realizations = np.ones([1,number_of_new_objects], dtype=int)

        # Check if there are new classes.
        no_new_classes_flag = False
        if cls_realizations.size == 0:
            cls_realizations = np.array([0])
            no_new_classes_flag = True

        # Updating gaussians; creating an updated belief matrix that eventually will replace the old one
        updated_belief_matrix = dict()

        relevancy_dict = dict()

        for prior_gaussian in self.belief_matrix:
            for cls_real in cls_realizations:

                # If no new objects observed
                if no_new_classes_flag is True:
                    tuple_collector = list(prior_gaussian[0])

                else:
                # First time step with objects; create a class realization
                    if prior_gaussian[0][0] is 0:
                        tuple_collector = list()
                        running_index = 0
                        for obj in self.obj_realization:
                            tuple_collector.append((obj, cls_real[running_index]))
                            running_index += 1

                    # If it's not the first time
                    else:
                        tuple_collector = list(prior_gaussian[0])
                        running_index = 0
                        for obj in self.obj_realization:
                            if self.obj_realization[obj] is 'new':
                                tuple_collector.append((obj, cls_real[running_index]))
                                running_index += 1

                # Collect all to class index
                cls_index = tuple(tuple_collector)

                # Creating new keys for the updated belief matrix
                new_da = list()
                new_da.append(cls_index)
                if len(prior_gaussian) > 0:
                    for partial_da_real in prior_gaussian[1:]:
                        new_da.append(partial_da_real)
                da_real = tuple(da_current_step)
                new_da.append(da_real)
                new_da = tuple(new_da)

                # Checking if class changes are relevant to current DA realization
                cls_relevancy = list()
                for i in self.obj_realization:
                    # Go over objects seen by main robot
                    flag_check = False
                    for each in new_da[1:]:
                        if i in each and flag_check is False:
                            # Find the obj class realization
                            obj_tuple = self.find_object_cls_in_realization(new_da[0], i)
                            cls_relevancy.append(obj_tuple)
                            flag_check = True
                relevancy_index = tuple([tuple(cls_relevancy), new_da[1:]])

                # If class difference between realizations isn't within the DA realization, the weight is cloned and
                # the computational effort significantly reduced
                current_cls_realization = self.extract_cls_realization(da_current_step, new_da)

                if relevancy_index not in relevancy_dict:
                    self.belief_matrix[prior_gaussian] = self.belief_matrix[prior_gaussian].clone()
                    updated_belief_matrix[new_da] = self.belief_matrix[prior_gaussian].clone()
                    updated_belief_matrix[new_da].add_measurements(geo_measurements, sem_measurements, da_current_step,
                                                                   new_da[0],
                                                                   number_of_samples=number_of_samples,
                                                                   new_input_object_prior=new_input_object_prior,
                                                                   new_input_object_covariance=new_input_object_covariance,
                                                                   ML_update=self.ML_update)
                    relevancy_dict[relevancy_index] = updated_belief_matrix[new_da].logWeight
                else:
                    updated_belief_matrix[new_da] = self.belief_matrix[prior_gaussian].clone()
                    updated_belief_matrix[new_da].add_measurements(geo_measurements, sem_measurements, da_current_step,
                                                                   new_da[0],
                                                                   number_of_samples=number_of_samples,
                                                                   weight_update_flag=False,
                                                                   new_input_object_prior=new_input_object_prior,
                                                                   new_input_object_covariance=new_input_object_covariance,
                                                                   ML_update=self.ML_update)
                    updated_belief_matrix[new_da].logWeight = relevancy_dict[relevancy_index]


                # Copy previous belief, the old one to be deleted later
                if updated_belief_matrix[new_da].logWeight == -math.inf:
                    del updated_belief_matrix[new_da]

        # print(relevancy_dict)
        # Override the belief matrix; prune and normalize the weights
        del self.belief_matrix
        self.belief_matrix = updated_belief_matrix
        empty_flag = self.weight_pruning_basic()

        # Empty belief check, required for fault management when doing multiple runs
        if empty_flag is False:
            print('Empty belief')
            self.empty_belief_flag = True
            return False

    # Convert class realization tuple to dictionary
    @staticmethod
    def convert_cls_to_dict(cls_real):

        converted = dict()
        for pair in cls_real:
            converted[pair[0]] = pair[1]

        return converted


    # Find object and class tuple in realization
    @staticmethod
    def find_object_cls_in_realization(realization, object):

        for obj in realization:
            if obj[0] is object:
                return obj
        return None

    # Extract current step class realization
    def extract_cls_realization(self, current_da, realization):

        cls_realization = list()
        for obj in current_da:
            for cls in realization[0]:
                if obj == cls[0]:
                    cls_realization.append(cls[1])

        return tuple(cls_realization)

    # Extract all weights of realization to a vector
    def extract_log_weight(self):

        # Collect all weights into a vector
        log_weight_vector = list()
        for realization in self.belief_matrix:
            log_weight_vector.append(self.belief_matrix[realization].logWeight)
        log_weight_vector = np.array(log_weight_vector)

        return log_weight_vector

    # Weight normalization method;
    def normalize_weights(self):

        log_weight_vector = self.extract_log_weight()

        # Multiply by a constant to avoid numerical issues and normalize
        log_weight_vector -= np.average(log_weight_vector)
        weight_vector = np.exp(log_weight_vector)
        weight_vector = weight_vector/sum(weight_vector)
        weight_vector += self.norm_epsilon
        weight_vector = weight_vector / sum(weight_vector)
        new_log_weight_vector = np.log(weight_vector)

        # Input new weights into the belief matrix
        i = 0
        for realization in self.belief_matrix:
            # updated_belief = self.belief_matrix[realization].clone()
            # updated_belief.logWeight = new_log_weight_vector[i]
            # updated_belief.weight_memory.append(new_log_weight_vector[i])
            # del self.belief_matrix[realization]
            # self.belief_matrix[realization] = updated_belief

            self.belief_matrix[realization].logWeight = new_log_weight_vector[i]
            # self.belief_matrix[realization].weight_memory.append(new_log_weight_vector[i])
            i += 1

        return new_log_weight_vector

    # Basic pruning scheme; set a threshold for distance from maximum weight and normalize accordingly
    def weight_pruning_basic(self):

        # Find maximum weight
        log_weight_vector = self.extract_log_weight()
        # print(log_weight_vector)
        if not list(log_weight_vector):
            return False
        else:
            max_weight = np.amax(log_weight_vector)
        # Remove realizations that do not satisfy the weight criteria
        updated_belief_matrix = dict()
        for realization in self.belief_matrix:
            if self.belief_matrix[realization].logWeight >= max_weight - self.pruning_threshold:
                updated_belief_matrix[realization] = self.belief_matrix[realization].clone()

        del self.belief_matrix
        self.belief_matrix = updated_belief_matrix

        # Normalize new weights and return them as a vector
        new_log_weight_vector = self.normalize_weights()
        return new_log_weight_vector

    # Computes cross entropy loss
    def cross_entropy_loss(self, class_GT, normalize=False):

        if self.empty_belief_flag is True:
            return np.inf

        # Collects all weights corresponding to the correct class
        cls_weights = self.compute_cls_weights()
        cls_weights_list = np.zeros(self.number_of_objects)
        object_recorder = list()
        for realization in cls_weights:
            for idx in range(self.number_of_objects):
                if realization[idx][1] == 1:
                    cls_weights_list[idx] += cls_weights[realization]
                object_recorder.append(realization[idx][0])
        CEL = 0
        if normalize is True:

            for idx in range(self.number_of_objects):

                if not object_recorder[idx]:
                    return np.inf

                if class_GT[object_recorder[idx]] == 1:
                    CEL -= np.log(cls_weights_list[idx]) / self.number_of_objects
                else:
                    CEL -= np.log(1 - cls_weights_list[idx]) / self.number_of_objects

        else:

            for idx in range(self.number_of_objects):
                if class_GT[object_recorder[idx]] == 1:
                    CEL -= np.log(cls_weights_list[idx])
                else:
                    CEL -= np.log(1 - cls_weights_list[idx])
            CEL -= np.log(0.5) * (len(class_GT) - self.number_of_objects)

        return CEL

    # def MSDE(self, class_GT, normalize = True):
    #
    #     if self.empty_belief_flag is True:
    #         return np.inf
    #
    #     # Collects all weights corresponding to the correct class
    #     cls_weights = self.compute_cls_weights()
    #     cls_weights_list = dict()
    #     for key in self.obj_realization:
    #         cls_weights_list[key] = np.zeros(self.number_of_classes)
    #     object_recorder = list()
    #     for idx in range(self.number_of_objects):
    #         for cls in range(self.number_of_classes):
    #             for realization in cls_weights:
    #                 if realization[idx][1] == cls + 1:
    #                     cls_weights_list[realization[idx][0]][cls] += cls_weights[realization]
    #                 object_recorder.append(realization[idx][0])
    #
    #     MSDE = 0
    #
    #     if normalize is True:
    #
    #         for obj in cls_weights_list:
    #             for cls_idx in range(self.number_of_classes):
    #
    #                 if not object_recorder[idx]:
    #                     return 1
    #
    #                 if class_GT[object_recorder[idx]] == cls_idx + 1:
    #                     MSDE += 1 / self.number_of_classes * (1 - cls_weights_list[obj][cls_idx]) ** 2 / self.number_of_objects
    #                 else:
    #                     MSDE += 1 / self.number_of_classes * (0 - cls_weights_list[obj][cls_idx]) ** 2 / self.number_of_objects
    #
    #     else:
    #
    #         for obj in cls_weights_list:
    #             for cls_idx in range(self.number_of_classes):
    #
    #                 if not object_recorder[idx]:
    #                     return 1
    #
    #                 if class_GT[object_recorder[idx]] == cls_idx + 1:
    #                     MSDE += 1 / self.number_of_classes * (1 - cls_weights_list[obj][cls_idx]) ** 2
    #                 else:
    #                     MSDE += 1 / self.number_of_classes * (0 - cls_weights_list[obj][cls_idx]) ** 2
    #
    #
    #     return MSDE

    def compute_beta_weights(self):

        # Extract the weight vector
        log_weight_vector = self.extract_log_weight()
        weight_vector = np.exp(log_weight_vector)

        # Create a new weight vector for beta realization only
        beta_weight_vector = dict()
        idx = 0
        for realization in self.belief_matrix:
            if realization[1:] in beta_weight_vector:
                beta_weight_vector[realization[1:]] += weight_vector[idx]
            else:
                beta_weight_vector[realization[1:]] = weight_vector[idx]
            idx += 1

        return beta_weight_vector

    def compute_beta_only_realizations(self):

        expectation_last = dict()
        expectation_flag = dict()
        for realization in self.belief_matrix:
            if realization[1:] not in expectation_flag:
                expectation_flag[realization[1:]] = True
                expectation_last[realization] = self.belief_matrix[realization].result.atPose3 \
                    (self.belief_matrix[realization].X(len(realization) - 1))

        return expectation_last

    def compute_cls_weights(self):

        # Extract the weight vector
        log_weight_vector = self.extract_log_weight()
        weight_vector = np.exp(log_weight_vector)

        # Create a new weight vector for beta realization only
        cls_weight_vector = dict()
        idx = 0
        for realization in self.belief_matrix:
            if realization[0] in cls_weight_vector:
                cls_weight_vector[realization[0]] += weight_vector[idx]
            else:
                cls_weight_vector[realization[0]] = weight_vector[idx]
            idx += 1

        return cls_weight_vector

    # Printing function of realizations and their weights
    def print_realization_weights(self):

        for realization in self.belief_matrix:
            i = 0
            for index in realization:
                if i == 0:
                    print('The class realization is: ' + str(index))
                    i += 1
                else:
                    print('The data association realization for time step k=' + str(i) + ' is: ' + str(index))
                    i += 1
            print('The weight of the realization is: ' + str(math.exp(self.belief_matrix[realization].logWeight)) +
                  '\n-----------------------------------------------------------------------\n')

    def graph_largest_weight(self, fig_num=0, plt_size=10):

        max_weight = -np.inf
        for realization in self.belief_matrix:
            if self.belief_matrix[realization].logWeight > max_weight:
                max_weight = self.belief_matrix[realization].logWeight
                max_realization = realization

        self.belief_matrix[max_realization].display_graph(fig_num, plt_size)

    def covariance_largest_weight(self, extract_latest_covariance_det = True, print_matrices=False):

        if self.empty_belief_flag is True:
            return np.inf

        max_weight = -np.inf
        for realization in self.belief_matrix:
            if self.belief_matrix[realization].logWeight > max_weight:
                max_weight = self.belief_matrix[realization].logWeight
                max_realization = realization

        if not self.belief_matrix:
            return None
        if print_matrices is True:
            self.belief_matrix[max_realization].print_covariance()
        if extract_latest_covariance_det is True:
            return np.linalg.det(self.belief_matrix[max_realization].marginals.marginalCovariance
                                 (self.belief_matrix[max_realization].X(len(max_realization)-1)))

    def covariance_det_all_objects_largest_weight(self, return_obj=True, print_matrices=False):

        if self.empty_belief_flag is True:
            return np.inf

        max_weight = -np.inf
        for realization in self.belief_matrix:
            if self.belief_matrix[realization].logWeight > max_weight:
                max_weight = self.belief_matrix[realization].logWeight
                max_realization = realization

        if not self.belief_matrix:
            return None
        if print_matrices is True:
            self.belief_matrix[max_realization].print_covariance()
        if return_obj is True:

            obj_average_cov = 0
            number_of_objects = len(self.obj_realization)

            for obj in self.obj_realization:
                obj_average_cov += np.linalg.det(self.belief_matrix[max_realization].\
                    marginals.marginalCovariance(self.belief_matrix[max_realization].XO(obj))[3:5,3:5]) \
                                   / number_of_objects

            return obj_average_cov

    def distance_largest_weight(self, GT_poses):

        if self.empty_belief_flag is True:
            return np.inf

        # Find the largest weight realization
        max_weight = -np.inf
        for realization in self.belief_matrix:
            if self.belief_matrix[realization].logWeight > max_weight:
                max_weight = self.belief_matrix[realization].logWeight
                max_realization = realization

        # Compute the distance sum for the last step
        # If this is the first step, optimize
        if len(max_realization) < 3:
            self.belief_matrix[max_realization].optimize()
        relative_pose = GT_poses[len(max_realization)-1].between(self.belief_matrix[max_realization].
                                            result.atPose3(self.belief_matrix[max_realization].
                                                           X(len(max_realization)-1)))
        distance = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2 + relative_pose.z()**2)

        return distance

    def distance_all_weights(self, GT_poses):
    # Compute the weighted distance average of the last step from ground truth

        if self.empty_belief_flag is True:
            return np.inf

        distance_weighted_sum = 0
        log_weight = self.extract_log_weight()
        weight_vector = np.exp(log_weight)
        idx = 0
        for realization in self.belief_matrix:
            last_step = len(realization)-1
            self.belief_matrix[realization].optimize()
            relative_pose = GT_poses[last_step].between(self.belief_matrix[realization].
                                                        result.atPose3(self.belief_matrix[realization].X(last_step)))
            distance = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2 + relative_pose.z()**2)

            distance_weighted_sum += weight_vector[idx] * distance
            idx += 1

        return distance_weighted_sum

    def obj_distance_largest_weight(self, GT_poses):

        if self.empty_belief_flag is True:
            return np.inf

        # Find the largest weight realization
        max_weight = -np.inf
        for realization in self.belief_matrix:
            if self.belief_matrix[realization].logWeight > max_weight:
                max_weight = self.belief_matrix[realization].logWeight
                max_realization = realization

        # Compute the distance sum for the last step
        # If this is the first step, optimize
        if len(max_realization) < 3:
            self.belief_matrix[max_realization].optimize()

        distance = 0
        # Run over all objects
        for i in range(len(GT_poses)):
            relative_pose = GT_poses[i].between(self.belief_matrix[max_realization].
                                                                   result.atPose3(self.belief_matrix[max_realization].
                                                                                  XO(i + 1)))
            distance += np.sqrt(relative_pose.x() ** 2 + relative_pose.y() ** 2 + relative_pose.z() ** 2)

        distance = distance / (len(GT_poses) + 1)
        return distance

    def obj_distance_all_weights(self, GT_poses):

        if self.empty_belief_flag is True:
            return np.inf

        distance_weighted_sum = 0
        log_weight = self.extract_log_weight()
        weight_vector = np.exp(log_weight)
        idx = 0
        for realization in self.belief_matrix:
            last_step = len(realization) - 1
            self.belief_matrix[realization].optimize()
            distance = 0
            # Run over all objects that are known.
            object_counter = 0
            for i in range(len(GT_poses)):
                if self.belief_matrix[realization].result.exists(self.belief_matrix[realization].
                                                                   XO(i + 1)) is True:
                    relative_pose = GT_poses[i].between(self.belief_matrix[realization].
                                                        result.atPose3(self.belief_matrix[realization].
                                                                       XO(i + 1)))
                    distance += np.sqrt(relative_pose.x() ** 2 + relative_pose.y() ** 2 + relative_pose.z() ** 2)
                    object_counter += 1

            if object_counter == 0:
                distance = np.inf
            else:
                distance = distance / object_counter

            distance_weighted_sum += weight_vector[idx] * distance
            idx += 1

        return distance_weighted_sum

    def covariance_all_weights(self):

        if self.empty_belief_flag is True:
            return np.inf

        cov_total = 0
        log_weight = self.extract_log_weight()
        weight_vector = np.exp(log_weight)
        idx = 0
        for realization in self.belief_matrix:
            last_step = len(realization) - 1
            if last_step < 2:
                self.belief_matrix[realization].optimize()
            cov_det = 0
            # Run over all objects
            cov = self.belief_matrix[realization].marginals_after.marginalCovariance(
                self.belief_matrix[realization].X(last_step))
            cov_det = np.linalg.det(cov[3:5, 3:5])

            cov_total += weight_vector[idx] * cov_det
            idx += 1

        return cov_total

    def covariance_objects_all_weights(self):

        if self.empty_belief_flag is True:
            return np.inf

        cov_total = 0
        log_weight = self.extract_log_weight()
        weight_vector = np.exp(log_weight)
        idx = 0
        for realization in self.belief_matrix:
            last_step = len(realization) - 1
            self.belief_matrix[realization].optimize()
            cov_det = 0
            # Run over all objects
            for obj in self.obj_realization:
                cov = self.belief_matrix[realization].marginals_after.marginalCovariance(
                    self.belief_matrix[realization].XO(obj))
                cov_det = np.linalg.det(cov[3:5,3:5])

                cov_det += cov_det / (len(self.obj_realization))

            cov_total += weight_vector[idx] * cov_det
            idx += 1

        return cov_total

    def graph_all(self, min_fig_num=0):

        i = 0
        for realization in self.belief_matrix:
            self.belief_matrix[realization].display_graph(fig_num= min_fig_num + i)
            i += 1

    def print_weight_vectors(self, plot_graphs=False):

        if plot_graphs is True:
            fig_w, ax_w = plt.subplots(len(self.belief_matrix), sharex=True)
            #fig_w.suptitle('Weight graphs for different realizations')

        i = 0
        for realization in self.belief_matrix:
            np_log_weights = np.array(self.belief_matrix[realization].weight_memory)
            np_weights = np.exp(np_log_weights)
            print('Class realization: ' + str(realization[0]) + ', DA realization: ' + str(realization[1:]) +
                  ', weight vector: ' + str(np_weights))
            if plot_graphs is True:
                if len(self.belief_matrix) == 1:
                    self.belief_matrix[realization].weight_graph(fig_w, ax_w)
                else:
                    self.belief_matrix[realization].weight_graph(fig_w, ax_w[i])
            i += 1

        if plot_graphs is True:
            plt.tight_layout()
            plt.show()

    def graph_realizations_in_one(self, key_num, fig_num=0, show_obj=False,
                                  show_weights=False, show_plot=True, plt_save_name=None):

        if show_weights is False:
            fig, ax = plt.subplots(1, sharex=False)
            ax_first = ax
        else:
            ax = list()
            fig= plt.figure(fig_num)
            grid = plt.GridSpec(1, 2)
            ax.append(fig.add_subplot(grid[0, 0]))
            ax.append(fig.add_subplot(grid[0, 1]))
            ax_first = ax[0]
        plt.gca()
        plt.rcParams.update({'font.size': 16})

        #fig = plt.figure(fig_num)
        #ax = fig.gca()
        #plt.gca()

        for realization in self.belief_matrix:
            self.belief_matrix[realization].optimize()

        #beta_realizations = self.compute_beta_only_realizations()
        #beta_weights = self.compute_beta_weights()

        idx = 0

        for realization in self.belief_matrix:


            marginals = self.belief_matrix[realization].marginals_after
            #marginals = gtsam.Marginals(self.belief_matrix[realization].graph, self.belief_matrix[realization].result)
            pose_exp = self.belief_matrix[realization].result.atPose3(self.belief_matrix[realization].X(key_num))
            pose_cov = marginals.marginalCovariance(
                self.belief_matrix[realization]
                .X(key_num))
            pose_cov_2x2 = pose_cov[3:5, 3:5]

            weight = np.exp(self.belief_matrix[realization].logWeight)

            plotterdaac2d.plotPose2(fig_num, pose_exp, ax=ax_first)
            #pose_cov_rotated = pose_cov_2x2
            pose_cov_rotated = self.belief_matrix[realization].rotate_cov_6x6(pose_cov, pose_exp.rotation().matrix())
            #pose_cov_rotated = plotterdaac2d.rotate_covariance(pose_exp, pose_cov_2x2)
            color_tuple = ((1 - weight) * 0.7,
                           (1 - weight) * 0.7,
                           (1 - weight) * 0.7)
            # Adjusting ellipse alpha values based on if classification is turned on or not.
            if self.cls_enable is False:
                line_thickness = weight* 3.3 + 0.2
            else:
                line_thickness = weight * 3.3 + 0.2
            if show_weights is False:
                plotterdaac2d.plot_ellipse(ax, [pose_exp.x(), pose_exp.y()], pose_cov_rotated[3:5,3:5], color=None,
                                           edgecolor='r',
                                           alpha=1, enlarge=3,
                                           linewidth=line_thickness)
                #ax.text(pose_exp.x(), pose_exp.y(), 'R' + str(idx+1))
            else:
                plotterdaac2d.plot_ellipse(ax[0], [pose_exp.x(), pose_exp.y()], pose_cov_rotated[3:5,3:5], color=None,
                                           edgecolor='r',
                                           alpha=1, enlarge=3,
                                           linewidth=line_thickness)
                #ax[0].text(pose_exp.x(), pose_exp.y(), 'R' + str(idx+1))

            if show_obj is True:
                for obj in self.obj_realization:
                    pose_obj = self.belief_matrix[realization].result.atPose3(self.belief_matrix[realization].XO(obj))
                    pose_obj_cov = self.belief_matrix[realization].marginals_after.marginalCovariance\
                        (self.belief_matrix[realization].XO(obj))
                    pose_obj_cov_2x2 = pose_obj_cov[3:5, 3:5]
                    plotterdaac2d.plotPoint2(fig_num, pose_obj, 'k.', ax=ax_first)
                    #pose_obj_cov_rotated = pose_obj_cov_2x2
                    pose_obj_cov_rotated = self.belief_matrix[realization].rotate_cov_6x6(pose_obj_cov,
                                                                                          pose_obj.rotation().matrix())
                    color_tuple = ((1 - np.exp(self.belief_matrix[realization].logWeight)) * 0.7,
                                (1 - np.exp(self.belief_matrix[realization].logWeight)) * 0.7,
                                (1 - np.exp(self.belief_matrix[realization].logWeight)) * 0.7)
                    line_thickness = np.exp(self.belief_matrix[realization].logWeight) * 2.5 + 0.5


                    if show_weights is False:
                        plotterdaac2d.plot_ellipse(ax, [pose_obj.x(), pose_obj.y()], pose_obj_cov_rotated[3:5,3:5],
                                                   color=None,
                                                   edgecolor=(0.5,0.5,0.5),
                                                   alpha=1, enlarge=3,
                                                   linewidth=line_thickness)
                    else:
                        plotterdaac2d.plot_ellipse(ax[0], [pose_obj.x(), pose_obj.y()], pose_obj_cov_rotated[3:5,3:5],
                                                   color=None,
                                                   edgecolor=(0.5,0.5,0.5),
                                                   alpha=1, enlarge=3,
                                                   linewidth=line_thickness)

            idx += 1

        if show_weights is True:

            # Plot class realization probability bar
            cls_weights = self.compute_cls_weights()
            cls_weights_list = np.zeros(self.number_of_objects)
            x_labels = list()
            for realization in cls_weights:
                for idx in range(self.number_of_objects):
                    if realization[idx][1] == 1:
                        cls_weights_list[idx] += cls_weights[realization]
                    x_labels.append('Object ' + str(realization[idx][0]))
            plotterdaac2d.plot_weight_bars(ax[1], cls_weights_list, labels=x_labels, log_weight_flag=False)
            x = np.arange(self.number_of_objects)
            ax[1].set_xticks(x)
            ax[1].set_xticklabels(x_labels)

            # Rotate labes
            for tick in ax[1].get_xticklabels():
                tick.set_rotation(90)

            #ax[2].set_title('Probability of class 1, all objects')

        #ax_first.set_title('Multiple realizations of pose X' + str(key_num) + ', after measurements')
        if show_plot is True:
            plt.tight_layout()
            if plt_save_name is not None:
                plt.savefig(plt_save_name + '.eps', format='eps')
            plt.show()

        return ax


    def graph_realization_before_measurements(self, key_num, fig_num=0, show_prev=False, show_data=False):

        fig = plt.figure(fig_num)
        ax = fig.gca()
        plt.gca()
        plt.rcParams.update({'font.size': 16})

        for realization in self.belief_matrix:
            pose_exp = self.belief_matrix[realization].result.atPose3(self.belief_matrix[realization].X(key_num))
            if show_data is True:
                print(pose_exp)
            pose_cov = self.belief_matrix[realization].marginals.marginalCovariance(
                self.belief_matrix[realization]
                .X(key_num))
            pose_cov_2x2 = pose_cov[3:5, 3:5]
            plotterdaac2d.plotPose2(fig_num, pose_exp)
            pose_cov_rotated = plotterdaac2d.rotate_covariance(pose_exp, pose_cov_2x2)
            color_tuple = ((1 - np.exp(self.belief_matrix[realization].logWeight)) * 0.7,
                           (1 - np.exp(self.belief_matrix[realization].logWeight)) * 0.7,
                           np.exp(self.belief_matrix[realization].logWeight))
            alpha_check = np.exp(self.belief_matrix[realization].logWeight) * 0.3 + 0.3
            plotterdaac2d.plot_ellipse(ax, [pose_exp.x(), pose_exp.y()], pose_cov_rotated, color=color_tuple,
                                       alpha=alpha_check)

            if show_prev is True:
                pose_exp_prev = self.belief_matrix[realization].result.atPose3(self.belief_matrix[realization].X(key_num-1))
                pose_cov_prev = self.belief_matrix[realization].marginals.marginalCovariance(
                    self.belief_matrix[realization]
                        .X(key_num-1))
                pose_cov_2x2_prev = pose_cov_prev[3:5, 3:5]
                plotterdaac2d.plotPose2(fig_num, pose_exp_prev)
                if show_data is True:
                    print(pose_exp_prev)



            # ax.text(pose_exp.x(), pose_exp.y(), 'X' + str(key_num))# + ', ' + str(np.exp(self.belief_matrix[realization].logWeight)))

        #plt.title('Multiple realizations of pose X' + str(key_num) + ', before measurements')
        plt.tight_layout()
        plt.show()

    def weight_bar_graph(self, fig_num=0):

        fig = plt.figure(fig_num)
        ax = fig.gca()
        plt.gca()
        plt.rcParams.update({'font.size': 16})

        # Print last associations and present ellipse to corresponding bar graph
        i = 0
        beta_weights = self.compute_beta_weights()
        beta_weights_list = list()
        x_labels = list()
        for realization in beta_weights:
            beta_weights_list.append(beta_weights[realization])
            label_text = 'DA:'
            for data in realization:
                label_text += '\n' + str(data)
            x_labels.append(label_text)
            i += 1

        plotterdaac2d.plot_weight_bars(ax, beta_weights_list, labels=x_labels, log_weight_flag=False)
        # ax.set_title('Data association realization probability')

        plt.tight_layout()
        plt.show()

    def cls_bar_graph(self, fig_num = 0, show_plt=True, plt_save_name=None, pause_time=None, GT_classes=None):

        fig = plt.figure(fig_num)
        ax = fig.gca()
        plt.gca()
        plt.rcParams.update({'font.size': 16})

        number_of_bars = self.number_of_classes - 1

        cls_weights = self.compute_cls_weights()
        cls_weights_list = dict()
        for cls_idx in range(number_of_bars):
            cls_weights_list[cls_idx + 1] = np.zeros(self.number_of_objects)
        x_labels = list()
        height = list()
        for realization in cls_weights:
            for idx in range(self.number_of_objects):
                if realization[idx][1] != self.number_of_classes:
                    cls_weights_list[realization[idx][1]][idx] += cls_weights[realization]
                x_labels.append('Object ' + str(realization[idx][0]))

                if GT_classes is not None:
                    if GT_classes[realization[idx][0]] == 1:
                        height.append(1)
                    if GT_classes[realization[idx][0]] == 2:
                        height.append(0)


        for cls_idx in range(number_of_bars):
            if cls_idx == 0:
                bars = plotterdaac2d.plot_weight_bars(ax, cls_weights_list[cls_idx + 1], labels=x_labels,
                                                      log_weight_flag=False)
            else:
                bars = plotterdaac2d.plot_weight_bars(ax, cls_weights_list[cls_idx + 1], labels=x_labels,
                                                      log_weight_flag=False, bottom=cls_weights_list[cls_idx])
        x = np.arange(self.number_of_objects)
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

    def ATE_largest_weight(self, GT_poses):

        if self.empty_belief_flag is True:
            return np.inf

        # Find the largest weight realization
        max_realization = None
        max_weight = -np.inf
        for realization in self.belief_matrix:
            if self.belief_matrix[realization].logWeight > max_weight:
                max_weight = self.belief_matrix[realization].logWeight
                max_realization = realization

        # If belief is empty, return nothing
        if max_realization is None:
            return None

        # Compute the distance sum for the last step
        # If this is the first step, optimize
        if len(max_realization) < 3:
            self.belief_matrix[max_realization].optimize()

        # Set up lists for alignment
        xyz_first = list()
        xyz_second = list()

        # Run over all objects

        for i in range(len(GT_poses)):

            xyz_first.append(np.array([GT_poses[i].x(), GT_poses[i].y(), GT_poses[i].z()]))

            estimated_pose_key = self.belief_matrix[max_realization].X(i)
            estimated_pose = self.belief_matrix[max_realization].result.atPose3(estimated_pose_key)
            xyz_second.append(np.array([estimated_pose.x(), estimated_pose.y(), estimated_pose.z()]))

        xyz_first = np.array(xyz_first)
        xyz_second = np.array(xyz_second)

        rot, trans, trans_error = self.align(np.transpose(xyz_first), np.transpose(xyz_second))

        distance = np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))
        return distance

    def ATE_all_weights(self, GT_poses):
    # Compute the weighted distance average of the last step from ground truth

        if self.empty_belief_flag is True:
            return np.inf

        distance_weighted_sum = 0

        log_weight = self.extract_log_weight()
        weight_vector = np.exp(log_weight)
        idx = 0
        for realization in self.belief_matrix:

            # Set up lists for alignment
            xyz_first = list()
            xyz_second = list()

            for i in range(len(GT_poses)):
                xyz_first.append(np.array([GT_poses[i].x(), GT_poses[i].y(), GT_poses[i].z()]))

                estimated_pose_key = self.belief_matrix[realization].X(i)
                estimated_pose = self.belief_matrix[realization].result.atPose3(estimated_pose_key)
                xyz_second.append(np.array([estimated_pose.x(), estimated_pose.y(), estimated_pose.z()]))

            xyz_first = np.array(xyz_first)
            xyz_second = np.array(xyz_second)

            rot, trans, trans_error = self.align(np.transpose(xyz_first), np.transpose(xyz_second))

            distance = np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))

            distance_weighted_sum += weight_vector[idx] * distance
            idx += 1

        return distance_weighted_sum

    @staticmethod
    def align(model, data):
        """Align two trajectories using the method of Horn (closed-form).

        Input:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

        Output:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

        """
        data_length = model.size / 3
        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - np.tile(np.reshape(model.mean(1), [3, 1]), int(data_length))
        data_zerocentered = data - np.tile(np.reshape(data.mean(1), [3, 1]), int(data_length))

        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U * S * Vh
        trans = data.mean(1) - np.dot(rot, model.mean(1))

        model_aligned = rot * model + np.transpose(trans)
        alignment_error = model_aligned - data

        trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error

    def record_belief(self, filename=str()):

        f = open('History/' + filename + 'weights', 'w')
        for realization in self.belief_matrix:

            name_str = str()
            for vector in realization:
                for value in vector:
                    name_str += str(value)
                name_str += '_'

            self.belief_matrix[realization].record_belief('History/' + filename + str(name_str))
            f.write(str(np.exp(self.belief_matrix[realization].logWeight)) + '\n')

        f.close()

    def covariance_objects_ind_GT(self, GT_cls_realization):

        if self.empty_belief_flag is True:
            return np.inf

        log_weight = self.extract_log_weight()
        weight_vector = np.exp(log_weight)
        idx = 0

        for realization in self.belief_matrix:
            if set(realization[0]).issubset(GT_cls_realization):
                break

        last_step = len(realization) - 1
        if last_step < 2:
            self.belief_matrix[realization].optimize()
        cov_det = 0
        # Run over all objects
        for obj in self.obj_realization:
            cov = self.belief_matrix[realization].marginals_after.marginalCovariance(
                self.belief_matrix[realization].XO(obj))
            cov_det = np.linalg.det(cov[3:5,3:5])

            cov_det += cov_det / (len(self.obj_realization))

        idx += 1

        return cov_det

    def MSDE(self, class_GT, normalize = True):

        if self.empty_belief_flag is True:
            return np.inf

        # Collects all weights corresponding to the correct class
        object_recorder = list()
        cls_weights = self.compute_cls_weights()
        cls_weights_list = dict()
        for cls_idx in range(self.number_of_classes):
            cls_weights_list[cls_idx + 1] = np.zeros(self.number_of_objects)
        for realization in cls_weights:
            for idx in range(self.number_of_objects):
                cls_weights_list[realization[idx][1]][idx] += cls_weights[realization]
                if realization[idx][0] not in object_recorder:
                    object_recorder.append(realization[idx][0])

        MSDE = 0

        if normalize is True:

            for cls_idx in cls_weights_list:
                for obj in range(self.number_of_objects):

                    if not object_recorder[obj]:
                        return 1

                    if class_GT[object_recorder[obj]] == int(cls_idx):
                        MSDE += 1 / self.number_of_classes * (1 - cls_weights_list[cls_idx][obj]) ** 2 / self.number_of_objects
                    else:
                        MSDE += 1 / self.number_of_classes * (0 - cls_weights_list[cls_idx][obj]) ** 2 / self.number_of_objects

        else:

            for cls_idx in cls_weights_list:
                for obj in range(self.number_of_objects):

                    if not object_recorder[obj]:
                        return self.number_of_objects

                    if class_GT[object_recorder[obj]] == int(cls_idx):
                        MSDE += 1 / self.number_of_classes * (1 - cls_weights_list[cls_idx][obj]) ** 2
                    else:
                        MSDE += 1 / self.number_of_classes * (0 - cls_weights_list[cls_idx][obj]) ** 2

        return MSDE