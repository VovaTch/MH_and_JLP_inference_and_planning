from __future__ import print_function
import numpy as np
import scipy as sp
from scipy import stats
import gtsam
from gtsam.symbol_shorthand import X, O, L
import math
import copy
import matplotlib.pyplot as plt
import gtsam_utils
import plotterdaac2d
import lambda_prior_factor
from mpl_toolkits.mplot3d import Axes3D

class JLPBelief:

    daRealization = []
    resultsFlag = 0

    # LAMBDA MUST BE INSERTED AS POINT 2 FOR NOW
    def __init__(self, geo_model, da_model, lambda_model, prior_mean, prior_noise,
                 lambda_prior_mean=(0., 0.), lambda_prior_noise=((0.5, 0.), (0., 0.5)), cls_enable=True):

        # Symbol initialization
        self.X = lambda i: X(i)
        self.XO = lambda j: O(j)
        self.Lam = lambda k: L(k)

        # Enable Lambda inference
        self.cls_enable = cls_enable

        # Camera pose prior
        self.graph = gtsam.NonlinearFactorGraph()
        self.graph.add(gtsam.PriorFactorPose3(self.X(0), prior_mean, prior_noise))

        # Setting initial values
        self.initial = gtsam.Values()
        self.initial.insert(self.X(0), prior_mean)
        self.prev_step_camera_pose = prior_mean
        self.daRealization = list()

        # Setting up models
        self.geoModel = geo_model
        self.daModel = da_model
        self.lambdaModel = lambda_model

        # Setting up ISAM2
        params2 = gtsam.ISAM2Params()
        #params2.relinearize_threshold = 0.01
        #params2.relinearize_skip = 1
        self.isam = gtsam.ISAM2(params2)
        self.isam.update(self.graph, self.initial)

        # Set custom lambda
        if type(lambda_prior_mean) is np.ndarray:
            self.lambda_prior_mean = lambda_prior_mean
        else:
            self.lambda_prior_mean = np.array(lambda_prior_mean)
        self.lambda_prior_cov = gtsam.noiseModel.Gaussian.Covariance(np.matrix(lambda_prior_noise))

        self.num_cls = len(self.lambda_prior_mean) + 1


        # Initialize object last lambda database
        self.object_lambda_dict = dict()

    # Clone method; TODO: see if more stuff are needed to clone, change to __copy__ constructor if needed
    def clone(self):
        """

        :type new_object: GaussianBelief
        """
        new_object = copy.copy(self)
        new_object.graph = self.graph.clone()
        new_object.daRealization = self.daRealization.copy()
        params3 = gtsam.ISAM2Params()
        # params3.relinearize_threshold = 0.01
        # params3.relinearize_skip = 1
        new_object.isam = gtsam.ISAM2(params3)
        new_object.isam.update(self.graph, self.initial)
        new_object.initial = gtsam.Values()
        for key in self.initial.keys():
            try:
                new_object.initial.insert(key, self.initial.atPose3(key))
            except:
                new_object.initial.insert(key, self.initial.atVector(key))

        new_object.object_lambda_dict = self.object_lambda_dict.copy()

        return new_object

    # Forwarding the belief
    def action_step(self, action, action_noise):

        graph_add = gtsam.NonlinearFactorGraph()
        initial_add = gtsam.Values()

        # Setting path and initiating DA realization.
        path_length = len(self.daRealization)
        self.daRealization.append(0)

        # Advancing the belief by camera
        self.graph.add(gtsam.BetweenFactorPose3(self.X(path_length), self.X(path_length + 1), action, action_noise))
        graph_add.add(gtsam.BetweenFactorPose3(self.X(path_length), self.X(path_length+1), action, action_noise))

        # Setting initial values, save an initial value vector for the propagated pose, need it later
        if self.initial.exists(self.X(path_length + 1)) is False:
            self.initial.insert(self.X(path_length + 1), self.initial.atPose3(self.X(path_length)).compose(action))
            # self.initial.insert(self.X(path_length + 1), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))
        if initial_add.exists(self.X(path_length + 1)) is False:
            initial_add.insert(self.X(path_length+1), self.initial.atPose3(self.X(path_length)).compose(action))
            # initial_add.insert(self.X(path_length + 1), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))

        self.isam.update(graph_add, initial_add)
        self.optimize()

        # Setting resultsFlag to false to require optimization
        self.resultsFlag = False

    def add_measurements(self, geo_measurements, sem_measurement_exp, sem_measurement_cov, da_current_step,
                         new_input_object_prior=None, new_input_object_covariance=None,
                         lambda_prior_mean=None, lambda_prior_cov=None):

        graph_add = gtsam.NonlinearFactorGraph()
        initial_add = gtsam.Values()

        for obj in da_current_step:

            obj = int(obj)

            if new_input_object_prior is not None:
                new_object_prior = new_input_object_prior[obj]
                self.graph.add(gtsam.PriorFactorPose3(self.XO(obj), new_object_prior, new_input_object_covariance))
                graph_add.add(gtsam.PriorFactorPose3(self.XO(obj), new_object_prior, new_input_object_covariance))
            else:
                new_object_prior = gtsam.Pose3(gtsam.Pose2(0., 0., 0.))

            if lambda_prior_mean is not None:
                new_lambda_prior_mean = lambda_prior_mean[obj]
                new_lambda_prior_cov = lambda_prior_cov[obj]
            else:
                new_lambda_prior_mean = self.lambda_prior_mean
                new_lambda_prior_cov = self.lambda_prior_cov

            if self.initial.exists(self.XO(obj)) is False:
                self.initial.insert(self.XO(obj), new_object_prior)

                self.graph.add(lambda_prior_factor.LambdaPriorFactor(self.Lam(int(obj * 1e6 + len(self.daRealization)-1)),
                                                                      new_lambda_prior_mean, new_lambda_prior_cov))
                graph_add.add(lambda_prior_factor.LambdaPriorFactor(self.Lam(int(obj * 1e6 + len(self.daRealization)-1)),
                                                                      new_lambda_prior_mean, new_lambda_prior_cov))

                self.initial.insert(self.Lam(int(obj * 1e6 + len(self.daRealization)-1)),
                                    self.lambda_prior_mean)
                if initial_add.exists(self.XO(obj)) is False:
                    initial_add.insert(self.XO(obj), new_object_prior)
                    initial_add.insert(self.Lam(int(obj * 1e6 + len(self.daRealization)-1)),
                                       self.lambda_prior_mean)
                #if self.isam.value_exists(self.Lam(int(obj * 1e6 + len(self.daRealization)))) is False:
            initial_add.insert(self.Lam(int(obj * 1e6 + len(self.daRealization))),
                                   self.lambda_prior_mean)
            # Save object last lambda
            self.object_lambda_dict[obj] = int(obj * 1e6 + len(self.daRealization))

        # Geometric model inference
        self.geoModel.add_measurement(geo_measurements, da_current_step, self.graph, self.initial, self.daRealization)
        self.geoModel.add_measurement(geo_measurements, da_current_step, graph_add, self.initial,
                                      self.daRealization)

        # Lambda belief activation
        if self.cls_enable is True:
            self.lambdaModel.add_measurements(sem_measurement_exp, sem_measurement_cov, da_current_step,
                                              self.graph, self.initial, self.daRealization)
            self.lambdaModel.add_measurements(sem_measurement_exp, sem_measurement_cov, da_current_step,
                                              graph_add, self.initial, self.daRealization)

        # ISAM2 add
        self.isam.update(graph_add, initial_add)
        self.optimize()
        self.marginals_after = gtsam.Marginals(self.graph, self.result)

        # Setting resultsFlag to false to require optimization
        self.resultsFlag = False

    def optimize(self, graph_to_optimize=None): #TODO: Make ISAM2 work

        self.isam.update()
        self.result = self.isam.calculateEstimate()
        self.initial = self.result
        # self.initial.print()
        self.resultsFlag = True

    def print_results(self, show_entropy = False):

        # Optimizing if not done before
        if self.resultsFlag is False:
            self.optimize()

        # Printing results
        print("\nFactor Graph:\n-----------------\n\n")
        print(self.graph)
        print("\nInitial Estimate:\n----------------\n\n")
        print(self.initial)
        print("\nFinal Result:\n----------------\n\n")
        print(self.result)

        marginals = gtsam.Marginals(self.graph, self.result)
        if self.cls_enable is True:
            for obj in self.object_lambda_dict:
                print("\nLambda Expectation of object " + str(obj) + ":")
                print(self.result.atVector(self.Lam(self.object_lambda_dict[obj])))
                print("\nLambda Covariance of object " + str(obj) + ":")
                print(marginals.marginalCovariance(self.Lam(self.object_lambda_dict[obj])))
                print("\n")

                if show_entropy is True:
                    entropy_upper, entropy_lower = self.lambda_entropy_individual_bounds(self.object_lambda_dict[obj])
                    entropy_num = self.lambda_entropy_individual_numeric(self.object_lambda_dict[obj])
                    print("Entropy upper limit: " + str(entropy_upper))
                    print("Entropy numerical value: " + str(entropy_num))
                    print("Entropy lower limit: " + str(entropy_lower))

        print("\n**-----------------------------------------")

    # Present the individual entropy (3D vector, if found )
    def lambda_entropy_individual_bounds(self, key, entropy_lower_limit=None):

        marginals = gtsam.Marginals(self.graph, self.result)
        covariance = marginals.marginalCovariance(self.Lam(key))
        exp_point = self.result.atVector(self.Lam(key))
        #exp = np.array([exp_point[0], exp_point[1]])
        exp = exp_point
        num_of_classes = self.lambda_prior_mean.size + 1
        entropy_upper = 0.5 * np.log(np.linalg.det(2 * np.exp(1) * np.pi * covariance)) \
                        + np.sum(exp) - num_of_classes * np.max(np.abs(exp))
        entropy_lower = entropy_upper - num_of_classes * np.log(num_of_classes)

        if entropy_lower_limit is not None and entropy_upper < entropy_lower_limit:
            entropy_upper = entropy_lower_limit
        if entropy_lower_limit is not None and entropy_lower < entropy_lower_limit:
            entropy_lower = entropy_lower_limit

        return entropy_upper, entropy_lower

    def lambda_entropy_individual_numeric(self, key, number_of_samples=20, entropy_lower_limit=None):

        marginals = gtsam.Marginals(self.graph, self.result)
        covariance = marginals.marginalCovariance(self.Lam(key))
        exp_point = self.result.atVector(self.Lam(key))
        #exp = np.array([exp_point[0], exp_point[1]])
        exp = exp_point

        for idx in range(len(exp)):
            if exp[idx] > 250:
                exp[idx] = 250
            if exp[idx] < -250:
                exp[idx] = 250

        entropy = 0
        for idx in range(number_of_samples):

            sample = np.random.multivariate_normal(exp, covariance)
            sample_aug = np.concatenate((sample, 0), axis=None)
            sample_cpv = np.exp(sample_aug)/(np.sum(np.exp(sample_aug)))
            log_pdf_value = np.log(stats.multivariate_normal.pdf(sample, exp, covariance)) - \
                        np.sum(np.log(sample_cpv))
            entropy -= log_pdf_value / number_of_samples

        if entropy_lower_limit is not None and entropy < entropy_lower_limit:
            entropy = entropy_lower_limit

        return entropy

    # Numerically compute the expectation of lambda (not lg-lambda), which is unknown how to compute analytically
    def lambda_expectation_numeric(self, key, number_of_samples=100):

        marginals = gtsam.Marginals(self.graph, self.result)
        covariance = marginals.marginalCovariance(self.Lam(key))
        exp_point = self.result.atVector(self.Lam(key))
        exp = exp_point

        exp_lambda = np.zeros(len(exp) + 1)

        for idx in range(number_of_samples):

            sample = np.random.multivariate_normal(exp, covariance)
            sample_aug = np.concatenate((sample, 0), axis=None)
            sample_cpv = np.exp(sample_aug)/(np.sum(np.exp(sample_aug)))
            exp_lambda += sample_cpv / number_of_samples

        return exp_lambda

    # Numerically compute the covariance of lambda, similarly to lambda_expectation_numeric,
    # denotes how wide the uncertainty bars in a bar graph.
    def lambda_covariance_numeric(self, key, number_of_samples=20):

        marginals = gtsam.Marginals(self.graph, self.result)
        covariance = marginals.marginalCovariance(self.Lam(key))
        exp_point = self.result.atVector(self.Lam(key))
        exp = exp_point

        exp_lambda = np.zeros(len(exp) + 1)
        cov_lambda = np.zeros((len(exp) + 1, len(exp) + 1))

        for idx in range(number_of_samples):

            sample = np.random.multivariate_normal(exp, covariance)
            sample_aug = np.concatenate((sample, 0), axis=None)
            sample_cpv = np.exp(sample_aug)/(np.sum(np.exp(sample_aug)))
            exp_lambda += sample_cpv / number_of_samples

        for idx in range(number_of_samples):

            sample = np.random.multivariate_normal(exp, covariance)
            sample_aug = np.concatenate((sample, 0), axis=None)
            sample_cpv = np.exp(sample_aug) / (np.sum(np.exp(sample_aug)))
            #print(sample_cpv)
            cov_lambda += np.outer(sample_cpv - exp_lambda, sample_cpv - exp_lambda) / number_of_samples

        return cov_lambda

    # Compute entropy of expectation of lambda
    def lambda_expectation_entropy(self, key, entropy_lower_limit=None):

        lambda_expectation = self.lambda_expectation_numeric(key)
        lambda_expectation += 1e-100

        entropy = 0

        for cls_idx in range(len(lambda_expectation)):

            entropy -= lambda_expectation[cls_idx] * np.log(lambda_expectation[cls_idx])

        if entropy_lower_limit is not None and entropy_lower_limit > entropy:
            entropy = entropy_lower_limit

        return entropy

    # MSDE of an object
    def MSDE_obj(self, obj, GT_class, number_of_samples=100):

        MSDE = 0

        lambda_expectation = self.lambda_expectation_numeric(self.object_lambda_dict[obj],
                                                             number_of_samples=number_of_samples)

        for cls_idx in range(len(self.lambda_prior_mean)):

            true_cls = 0
            if GT_class == cls_idx + 1:
                true_cls = 1

            MSDE += (lambda_expectation[cls_idx] - true_cls) ** 2 / len(self.lambda_prior_mean)

        return MSDE


    # Compute MSDE of expectation of lambda
    def MSDE_expectation(self, GT_realization, number_of_samples=100):

        MSDE = 0

        for obj in GT_realization:

            if obj in self.object_lambda_dict:
                lambda_expectation = self.lambda_expectation_numeric(self.object_lambda_dict[obj],
                                                                     number_of_samples=number_of_samples)
                for cls_idx in range(len(self.lambda_prior_mean) + 1):

                    true_cls = 0
                    if GT_realization[obj] == cls_idx + 1:
                        true_cls = 1

                    MSDE += (lambda_expectation[cls_idx] - true_cls) ** 2 / (len(self.lambda_prior_mean) + 1) \
                            / len(GT_realization)

            else:
                for cls_idx in range(len(self.lambda_prior_mean) + 1):

                    true_cls = 0
                    if GT_realization[obj] == cls_idx + 1:
                        true_cls = 1

                    MSDE += ( 1 / (len(self.lambda_prior_mean) + 1) - true_cls ) ** 2 / \
                            (len(self.lambda_prior_mean) + 1) / len(GT_realization)

        return MSDE


    # Generate measurements
    def generate_future_measurements(self, action, action_noise, geo_noise=np.eye(6), ML_geo=True, ML_cls=True):

        propagated_belief = self.clone()
        propagated_belief.action_step(action, action_noise)
        propagated_belief.optimize()

        geo_measurements = dict()
        sem_measurements_exp = dict()
        sem_measurements_cov = dict()

        if ML_geo is True:

            propagated_belief.optimize()
            sampled_robot_pose = propagated_belief.result.atPose3(propagated_belief.X
                                                                  (len(propagated_belief.daRealization)))
            for obj in propagated_belief.object_lambda_dict:

                # ML of estimated object pose
                ML_object_pose = propagated_belief.result.atPose3(propagated_belief.XO(obj))

                # Find if the object will be observed
                DA_flag = propagated_belief.daModel.object_observation_3d(sampled_robot_pose, ML_object_pose)
                if DA_flag is 1:

                    ML_llambda = propagated_belief.result.atVector(propagated_belief.Lam(
                        propagated_belief.object_lambda_dict[obj]))
                    ML_llambda = np.concatenate((ML_llambda, np.array([0])), axis=0)
                    if ML_cls is True:

                        # Find highest probability class
                        chosen_cls = 0
                        counter = 1
                        value_compare = 0
                        for value in ML_llambda:
                            if value >= value_compare:
                                value_compare = value
                                chosen_cls = counter
                                counter += 1

                    else:

                        # Randomly roll class WITH NUMBER OF CLASSES HARD CODED IN
                        ML_llambda_aug = np.concatenate((ML_llambda, 0), axis=None)
                        ML_lambda = np.exp(ML_llambda_aug) / np.sum(np.exp(ML_llambda_aug))
                        chosen_cls = np.random.choice(self.num_cls - 1, p=ML_lambda) + 1

                    #print(chosen_cls)

                    #geo_measurements[obj] = sampled_robot_pose.between(ML_object_pose)

                    geo_measurements[obj] = propagated_belief.geoModel.model_sample(sampled_robot_pose, ML_object_pose,
                                                                                    ML_sample=True)

                    relative_pose = ML_object_pose.between(sampled_robot_pose)
                    relative_position = [relative_pose.x(), relative_pose.y(), relative_pose.z()]

                    exp, cov = propagated_belief.lambdaModel.sample_lambda_model(chosen_cls, relative_position)
                    sem_measurements_exp[obj] = exp
                    sem_measurements_cov[obj] = cov

            sem_measurements = [sem_measurements_exp, sem_measurements_cov]
        else: #TODO: COMPLETE THE NON ML CASE

            propagated_belief.optimize()
            sampled_robot_pose = propagated_belief.result.atPose3(propagated_belief.X
                                                                  (len(propagated_belief.daRealization)))
            for obj in propagated_belief.object_lambda_dict:

                # ML of estimated object pose
                ML_object_pose = propagated_belief.result.atPose3(self.XO(obj))

                # Find if the object will be observed
                DA_flag = propagated_belief.daModel.object_observation_3d(sampled_robot_pose, ML_object_pose)
                if DA_flag is 1:

                    ML_llambda_exp = propagated_belief.result.atVector(propagated_belief.
                                                                       Lam(propagated_belief.object_lambda_dict[obj]))
                    ML_llambda_cov = propagated_belief.marginals_after.marginalCovariance\
                        (propagated_belief.Lam(propagated_belief.object_lambda_dict[obj]))
                    ML_llambda = np.random.multivariate_normal(ML_llambda_exp, ML_llambda_cov)
                    ML_llambda = np.concatenate((ML_llambda, np.array([0])), axis=0)
                    if ML_cls is True:

                        # Find highest probability class
                        chosen_cls = 0
                        counter = 1
                        value_compare = 0
                        for value in ML_llambda:
                            if value >= value_compare:
                                value_compare = value
                                chosen_cls = counter
                                counter += 1

                    else:

                        # Randomly roll class WITH NUMBER OF CLASSES HARD CODED IN
                        ML_lambda = np.exp(ML_llambda) / np.sum(np.exp(ML_llambda))
                        chosen_cls = np.random.choice(range(self.num_cls), p=ML_lambda) + 1

                    geo_measurements[obj] = propagated_belief.geoModel.model_sample\
                        (sampled_robot_pose, ML_object_pose, geo_noise)

                    zero_pose = gtsam.Pose3(gtsam.Pose2(0., 0., 0.))
                    relative_pose = geo_measurements[obj].between(zero_pose)
                    relative_position = [relative_pose.x(), relative_pose.y(), relative_pose.z()]

                    exp, cov = propagated_belief.\
                        lambdaModel.sample_lambda_model(chosen_cls, relative_position)
                    sem_measurements_exp[obj] = exp
                    sem_measurements_cov[obj] = cov

            sem_measurements = [sem_measurements_exp, sem_measurements_cov]

        return geo_measurements, sem_measurements

    def find_latest_key(self, obj):

        for step in range(len(self.daRealization), -1, -1):
            if self.result.exists(self.Lam(int(1e6 * obj + step))):
                return int(1e6 * obj + step)


    # TODO: Create this class for visualization
    def simplex_3class(self, selected_object, show_plot=True, write_lambdas=True, log_likelihood=False):

        fig = plt.figure(0)
        ax = fig.gca()

        expectation = self.result.atVector(self.Lam(self.find_latest_key(selected_object)))
        covariance = self.marginals_after.marginalCovariance(self.Lam(self.find_latest_key(selected_object)))

        if write_lambdas is True:
            print('Mu: ' + str(expectation) + '; \nCovariance: \n' + str(covariance))

        # x1_array = np.arange(1e-4, 1e-3, 1e-4)
        # x1_array = np.concatenate((x1_array, np.arange(1e-4, 1-1e-4, 1e-3)), axis=0)
        # x1_array = np.concatenate((x1_array, np.arange(1-1e-4, 1-1e-5, 1e-5)), axis=0)
        # x2_array = np.arange(1e-4, 1e-3, 1e-4)
        # x2_array = np.concatenate((x2_array, np.arange(1e-4, 1-1e-4, 1e-3)), axis=0)
        # x2_array = np.concatenate((x2_array, np.arange(1-1e-4, 1-1e-5, 1e-5)), axis=0)
        x1_array = np.arange(1e-2, 1-1e-2, 1e-2)
        x2_array = np.arange(1e-2, 1-1e-2, 1e-2)

        x1v, x2v = np.meshgrid(x1_array, x2_array)
        pdf_val = np.zeros((x1_array.size, x2_array.size))

        for idx_1 in range(x1_array.size):
            for idx_2 in range(x2_array.size):
                if x1_array[idx_1] + x2_array[idx_2] < 1:
                    querry = np.array([x1_array[idx_1], x2_array[idx_2], 1 - x1_array[idx_1] - x2_array[idx_2]])
                    querry_prod = 1 / (np.prod(querry))
                    lquerry = np.array([np.log(querry[0]/querry[2]), np.log(querry[1]/querry[2])])
                    pdf_val[idx_1, idx_2] = stats.multivariate_normal.pdf(lquerry, expectation, covariance) * \
                                            querry_prod
                    if log_likelihood is True:
                        pdf_val[idx_1, idx_2] = np.log(pdf_val[idx_1, idx_2])
                else:
                    pdf_val[idx_1, idx_2] = np.NaN

        ax.contourf(x1v, x2v, pdf_val, 200)#, cmap=plt.cm.Spectral)
        plt.xlabel('Probability of class 1')
        plt.ylabel('Probability of class 2')
        if show_plot is True:
            plt.show()

    # Function to display results in a figure
    def display_graph(self, fig_num=0, plt_size=10, plot_line = False, display_title = True, show_plot = True,
                      color = 'r', ax=None, enlarge=1):

        # Figure ID
        if ax is None:
            fig = plt.figure(fig_num)
            ax = fig.gca()
        plt.gca()

        if self.resultsFlag is False:
            self.optimize()
        plotterdaac2d.plot2DPoints(fig_num, self.result, 'rx')

        # Plot cameras
        if plot_line is False:
            idx = 0
            while self.result.exists(self.X(idx)):

                pose_x = self.result.atPose3(self.X(idx))
                plotterdaac2d.plotPose2(fig_num, pose_x, 0.3)

                # Annotate the point
                ax.text(pose_x.x(), pose_x.y(), 'X' + str(idx))

                # Plot covariance ellipse
                mu, cov = plotterdaac2d.extract_mu_cov_2d(self.result, self.marginals_after, self.X(idx))

                # Rotate covariance matrix
                # cov_rotated = plotterdaac2d.rotate_covariance(pose_x, cov)
                angles = pose_x.rotation().matrix()
                psi = np.arctan2(angles[1, 0], angles[0, 0])
                gRp = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
                multi1 = np.matmul(gRp, cov)
                cov_rotated = np.matmul(multi1, np.transpose(gRp))

                # Plot rotated cov ellipse
                plotterdaac2d.plot_ellipse(ax, mu, cov_rotated, enlarge=enlarge)

                idx += 1

        else:
            idx = 0
            pose_x_collector = list()
            pose_xo_collector = list()
            while self.result.exists(self.X(idx)):

                pose_x_collector.append(self.result.atPose3(self.X(idx)))

                idx += 1

            plotterdaac2d.GT_plotter_line(pose_x_collector, [], fig_num=0, ax=ax, show_plot=False,
                                          plt_save_name=None, pause_time=None,
                                          red_point=None, jpg_save=False, color=color, alpha=0.7)


        # Plot objects
        # self.result.print()
        for idx in self.object_lambda_dict:

            pose_xo = self.result.atPose3(self.XO(idx))
            pose_xo_cov = self.marginals_after.marginalCovariance(self.XO(idx))
            plotterdaac2d.plotPoint2(fig_num, pose_xo.translation(), 'go', ax=ax)

            # Annotate the point
            ax.text(pose_xo.x(), pose_xo.y(), 'O' + str(idx))

            # Plot covariance ellipse
            mu, cov = plotterdaac2d.extract_mu_cov_2d(self.result, self.marginals_after, self.XO(idx))

            # Rotate covariance matrix
            angles = pose_xo.rotation().matrix()
            psi = np.arctan2(angles[1, 0], angles[0, 0])
            gRp = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
            multi1 = np.matmul(gRp, cov)
            cov_rotated = np.matmul(multi1, np.transpose(gRp))

            # Plot rotated cov ellipse
            plotterdaac2d.plot_ellipse(ax, mu, cov_rotated, enlarge=enlarge)

        if display_title is True:
            plt.title('Figure number: ' + str(fig_num) + '. \nData association realization: ' + str(self.daRealization))
        ax.set_xlabel('X axis [m]')
        ax.set_ylabel('Y axis [m]')
        # ax.set_zlabel('Z axis [m]')

        # draw
        plt.tight_layout()
        if show_plot is True:
            plt.show()

        return ax

    # 2 class simplex figure
    def simplex_2class(self, selected_object, write_lambdas=True, show_plot=True, alpha_g=1):

        fig = plt.figure(0)
        ax = fig.gca()

        chosen_index = int(self.object_lambda_dict[selected_object])
        expectation = self.result.atVector(self.Lam(chosen_index))
        covariance = self.marginals_after.marginalCovariance(self.Lam(chosen_index))


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

    # Print all latest lambda parameters
    def print_lambda_lg_params(self):

        for obj_idx in self.object_lambda_dict.values():
            print("---------Object number " + str(obj_idx // 1e6) + ": --------")
            print(self.result.atVector(self.Lam(obj_idx)))
            print(self.marginals_after.marginalCovariance(self.Lam(obj_idx)))
            print("\n")

    # Distance metric computation
    def distance_metric(self, GT_poses_robot, GT_poses_objects):

        distance_robot = 0
        distance_object = 0

        for object in self.object_lambda_dict:
            relative_pose = GT_poses_objects[object].between(self.result.atPose3(self.XO(object)))
            distance_object += np.sqrt(relative_pose.x() ** 2 + relative_pose.y() ** 2 + relative_pose.z() ** 2) \
                               / len(self.object_lambda_dict)

        last_pose = len(self.daRealization)
        relative_pose = GT_poses_robot[last_pose].between(self.result.atPose3(self.X(last_pose)))
        distance_robot += np.sqrt(relative_pose.x() ** 2 + relative_pose.y() ** 2 + relative_pose.z() ** 2)

        return distance_robot, distance_object

    def display_graph_line(self):
        pass

    # Create bar graph with ranges
    def lambda_bar_error_graph(self, fig_num = 0, show_plt=True, plt_save_name=None, pause_time=None, GT_classes=None,
                               plot_individual_points=False, number_of_samples = 20, ax=None, color='blue', width_offset=0, GT_bar=False):

        object_list = self.object_lambda_dict

        # If no objects observed yet return 0
        if 'object_list' not in locals():
            return 0

        if ax is None:
            fig = plt.figure(fig_num)

            ax = fig.gca()
            plt.gca()
            plt.rcParams.update({'font.size': 16})

        number_of_bars = self.num_cls - 1
        number_of_objects = len(object_list)

        # Create plot-able data segments
        x_labels = list()
        height = list()
        cls_weights_list = dict()
        for cls_idx in range(self.num_cls - 1):

            cls_weights_list[cls_idx + 1] = np.zeros([number_of_samples, number_of_objects])

            for obj_num, obj in enumerate(object_list):
                exp = self.result.atVector(self.Lam(object_list[obj]))
                cov = self.marginals_after.marginalCovariance(self.Lam(object_list[obj]))

                cls_weights_list_lg = np.random.multivariate_normal(exp, cov, number_of_samples)
                for idx in range(number_of_samples):
                    cls_weights_list[cls_idx + 1][idx, obj_num] += np.divide(np.exp(cls_weights_list_lg[idx, cls_idx]),
                                                              (1 + np.sum(np.exp(cls_weights_list_lg[idx, :]))))

                x_labels.append('Object ' + str(obj))
                if GT_classes is not None:
                    if GT_classes[obj] == 1:
                        height.append(1)
                    if GT_classes[obj] == 2:
                        height.append(0)

        # If GT_bar is true, show bar relative to the GT class.
        if GT_bar is True:
            for obj_num, obj in enumerate(object_list):
                for cls_idx in range(number_of_bars):
                    if cls_idx + 1 != GT_classes[obj]:
                        cls_weights_list[cls_idx + 1][:, obj_num] = \
                            1 - cls_weights_list[cls_idx + 1][:, obj_num]


        # Plot the data
        for cls_idx in range(number_of_bars):
            if cls_idx == 0:
                bars = plotterdaac2d.plot_weight_bars_multi(ax, cls_weights_list[cls_idx + 1], labels=x_labels,
                                                            log_weight_flag=False,
                                                            plot_individual_points=plot_individual_points, color=color,
                                                            width_offset=width_offset)
            else:
                bars = plotterdaac2d.plot_weight_bars_multi(ax, cls_weights_list[cls_idx + 1], labels=x_labels,
                                                            log_weight_flag=False, bottom=cls_weights_list[cls_idx],
                                                            plot_individual_points=plot_individual_points, color=color,
                                                            width_offset=width_offset)
        x = np.arange(len(object_list))
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        # Rotate labes
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

        if GT_classes is not None and GT_bar is False:
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

    # Create bar graph with ranges
    def lambda_bar_error_graph_multi(self, fig_num=0, show_plt=True, plt_save_name=None, pause_time=None, GT_classes=None,
                               plot_individual_points=False, number_of_samples=20, ax=None, color='blue',
                               width_offset=0, GT_bar=False):

        object_list = self.object_lambda_dict

        # If no objects observed yet return 0
        if 'object_list' not in locals():
            return 0

        if ax is None:
            fig = plt.figure(fig_num)

            ax = fig.gca()
            plt.gca()
            plt.rcParams.update({'font.size': 16})

        number_of_bars = self.num_cls - 1
        number_of_objects = len(object_list)

        # Create plot-able data segments
        x_labels = list()
        height = list()
        cls_weights_list = dict()
        for cls_idx in range(self.num_cls - 1):

            cls_weights_list[cls_idx + 1] = np.zeros([number_of_samples, len(GT_classes)])

            for obj_num, obj in enumerate(GT_classes):

                if obj in object_list:
                    exp = self.result.atVector(self.Lam(object_list[obj]))
                    cov = self.marginals_after.marginalCovariance(self.Lam(object_list[obj]))
                else:
                    exp = np.ones(self.num_cls - 1) * -1e6
                    cov = np.diag(np.ones(self.num_cls - 1))

                cls_weights_list_lg = np.random.multivariate_normal(exp, cov, number_of_samples)
                for idx in range(number_of_samples):
                    cls_weights_list[cls_idx + 1][idx, obj_num] += np.divide(
                        np.exp(cls_weights_list_lg[idx, cls_idx]),
                        (1 + np.sum(np.exp(cls_weights_list_lg[idx, :]))))

                x_labels.append('Object ' + str(obj))
                if GT_classes is not None:
                    if GT_classes[obj] == 1:
                        height.append(1)
                    if GT_classes[obj] == 2:
                        height.append(0)

        print(x_labels)

        # If GT_bar is true, show bar relative to the GT class.
        cls_weights_list_gt = np.zeros([number_of_samples, len(GT_classes)])
        if GT_bar is True:
            for obj_num, obj in enumerate(GT_classes):
                for cls_idx in range(number_of_bars):
                    if cls_idx + 1 == GT_classes[obj]:
                        cls_weights_list_gt[:, obj_num] = cls_weights_list[cls_idx + 1][:, obj_num]

        cls_weights_dict = dict()
        for obj_num, obj in enumerate(object_list):
            cls_weights_dict[obj] = cls_weights_list_gt[:, obj_num]

        bars = plotterdaac2d.plot_weight_bars_multi(ax, cls_weights_list_gt, labels=None,
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

        # Rotate labes
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

        if GT_classes is not None and GT_bar is False:
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
