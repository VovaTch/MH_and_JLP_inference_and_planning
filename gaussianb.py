from __future__ import print_function
import numpy as np
import gtsam
import math
import copy
import matplotlib.pyplot as plt
import gtsam_utils
import plotterdaac2d
import copyreg
from mpl_toolkits.mplot3d import Axes3D



class GaussianBelief:

    daRealization = []
    logWeight = 0
    resultsFlag = False

    # Initiation of the belief, time 0. This is a belief for a specific DA and class realization
    def __init__(self, class_probability_prior, geo_model, da_model, cls_model,
                 prior_mean, prior_noise, cls_enable=True):

        # Symbol initialization
        self.X = lambda i: int(gtsam.Symbol('x', i))
        self.XO = lambda j: int(gtsam.Symbol('o', j))

        # Camera pose prior
        self.graph = gtsam.NonlinearFactorGraph()
        self.graph.add(gtsam.PriorFactorPose3(self.X(0), prior_mean, prior_noise))

        # Class realization and prior probabilities. if one of the class probabilities is 0 it will
        # set the logWeight to -inf
        self.cls_enable = cls_enable

        # Setting initial values
        self.initial = gtsam.Values()
        self.initial.insert(self.X(0), prior_mean)
        self.prev_step_camera_pose = prior_mean
        self.daRealization = list()

        # self.initial.print()

        # Setting up models
        self.geoModel = geo_model
        self.daModel = da_model
        self.clsModel = cls_model
        # self.classifierModel = classifierModel

        # Setting up ISAM2 TODO: make ISAM2 work. As of now, it doesn't take the initial values at optimize_isam2 method
        params = gtsam.ISAM2Params()
        # params.relinearization_threshold = 0.01
        params.relinearize_skip = 1
        self.isam = gtsam.ISAM2(params)

        # Setting up weight memory
        self.weight_memory = list()
        self.normalized_weight = 0

        # Setting up class realization
        self.classRealization = dict()
        self.classProbabilityPrior = class_probability_prior

    # Clone method; TODO: see if more stuff are needed to clone, change to __copy__ constructor if needed
    def clone(self):
        """

        :type new_object: GaussianBelief
        """

        new_object = copy.copy(self)
        new_object.graph = self.graph.clone()
        new_object.daRealization = self.daRealization.copy()
        new_object.classRealization = self.classRealization.copy()
        #new_object.isam = self.isam.copy()
        #new_object.result = self.result.copy()
        #new_object.marginals = self.marginals.copy()
        #new_object.marginals_after = self.marginals_after.copy()
        #new_object.initial = copy.copy(self.initial)
        #new_object.result = copy.copy(self.result)
        return new_object


    # Forwarding the belief
    def action_step(self, action, action_noise):

        # Setting path and initiating DA realization.
        path_length = len(self.daRealization)
        self.daRealization.append(0)

        # Advancing the belief by camera
        self.graph.add(gtsam.BetweenFactorPose3(self.X(path_length), self.X(path_length+1), action, action_noise))

        # Setting initial values, save an initial value vector for the propagated pose, need it later
        if self.initial.exists(self.X(path_length + 1)) is False:
            self.initial.insert(self.X(path_length + 1), self.initial.atPose3(self.X(path_length)).compose(action))

        # Setting propagated belief object for weight computation
        self.prop_belief = self.graph.clone()
        self.prop_initial = gtsam.Values()
        for key in list(self.initial.keys()):
            self.prop_initial.insert(key, self.initial.atPose3(key))

        # self.optimize() #TODO: SEE IF WORKS
        self.optimize(graph_to_optimize=self.prop_belief)

        # non-existent object issue is resolved.
        self.marginals = gtsam.Marginals(self.prop_belief, self.result)

        # Setting resultsFlag to false to require optimization
        self.resultsFlag = False

        # Multi robot flag for twin measurements
        self.sim_measurement_flag = list()

    # Sample a future pose given action and noise
    def motion_sample(self, action, action_noise_matrix, ML=False):

        # Taking the path length for using the latest pose
        try:
            self.optimize()
        except:
            pass
        path_length = len(self.daRealization)
        new_pose = self.result.atPose3(self.X(path_length)).compose(action)

        new_pose_vector = [new_pose.rotation().rpy()[0], new_pose.rotation().rpy()[1], new_pose.rotation().rpy()[2],
                           new_pose.x(), new_pose.y(), new_pose.z()]

        # Sample unless ML assumption is true
        if ML is False:

            sampled_new_pose_vector = np.random.multivariate_normal(new_pose_vector, action_noise_matrix)
            return gtsam.Pose3(gtsam.Rot3.Ypr(sampled_new_pose_vector[2], sampled_new_pose_vector[1],
                                              sampled_new_pose_vector[0]),
                               gtsam.Point3(sampled_new_pose_vector[3], sampled_new_pose_vector[4],
                                            sampled_new_pose_vector[5]))
        else:
            return new_pose

    # Adding measurements+ DA realization
    def add_measurements(self, geo_measurements, sem_measurements, da_current_step, class_realization_current_step,
                         number_of_samples=10, weight_update_flag=True,
                         new_input_object_prior=None, new_input_object_covariance=None, ML_update=False):

        # Setting up complete class realization
        running_index = 0
        for obj in da_current_step:
            if obj not in self.classRealization:

                # Create a class realization dict from the realization key
                for obj_index in class_realization_current_step:
                    if obj == obj_index[0]:
                        self.classRealization[obj] = obj_index[1]

                self.logWeight += np.log(self.classProbabilityPrior[self.classRealization[obj] - 1])

                # System is ill-defined without a prior for objects, so we define a prior.
                # Default is very uninformative prior
                new_object_prior = gtsam.Pose3(gtsam.Rot3.Ypr(0, 0, 0), gtsam.Point3(0, 0, 0))
                if new_input_object_prior is None:
                    pass
                    #new_object_prior = gtsam.Pose3(gtsam.Rot3.Ypr(0, 0, 0), gtsam.Point3(0, 0, 0))
                else:
                    new_object_prior = new_input_object_prior[obj]
                    self.graph.add(gtsam.PriorFactorPose3(self.XO(obj), new_object_prior, new_object_covariance))
                    self.prop_belief.add(gtsam.PriorFactorPose3(self.XO(obj), new_object_prior, new_object_covariance))

                if new_input_object_covariance is None:
                    pass
                    #new_object_covariance_diag = np.array([1000, 1000, 1000, 1000, 1000, 1000])
                    #new_object_covariance = gtsam.noiseModel.Diagonal.Variances(new_object_covariance_diag)
                else:
                    new_object_covariance = new_input_object_covariance[obj]
                    self.graph.add(gtsam.PriorFactorPose3(self.XO(obj), new_object_prior, new_object_covariance))
                    self.prop_belief.add(gtsam.PriorFactorPose3(self.XO(obj), new_object_prior, new_object_covariance))

                if self.initial.exists(self.XO(obj)) is False:
                    self.initial.insert(self.XO(obj), new_object_prior)
                    self.prop_initial.insert(self.XO(obj), new_object_prior)

            running_index += 1

        for obj in self.classRealization:
            if self.initial.exists(self.XO(obj)) is False:
                self.initial.insert(self.XO(obj), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))


        # Geometric measurement model
        self.geoModel.add_measurement(geo_measurements, da_current_step, self.graph, self.initial, self.daRealization)
        self.geoModel.add_measurement(geo_measurements, da_current_step, self.prop_belief, self.initial, self.daRealization)


        # Semantic measurement model
        if self.cls_enable is True:
            self.clsModel.add_measurement(sem_measurements, da_current_step, self.graph,
                                        self.initial, self.daRealization, class_realization_current_step)

        # Updating weights
        if weight_update_flag is True:
            if ML_update is False:
                self.weight_update(da_current_step, geo_measurements, sem_measurements, number_of_samples)
            else:
                self.weight_update_ML(da_current_step, geo_measurements, sem_measurements)

        self.optimize()
        self.marginals_after = gtsam.Marginals(self.graph, self.result)

        # Setting resultsFlag to false to require optimization
        self.resultsFlag = False

    def add_mr_object_measurement(self, stack, class_realization,
                                  new_input_object_prior=None, new_input_object_covariance=None):
        graph_add = gtsam.NonlinearFactorGraph()
        initial_add = gtsam.Values()

        idx = 0
        for obj in stack:

            cov_noise_model = stack[obj]['covariance']
            exp_gtsam = gtsam.Pose3(gtsam.Rot3.Ypr(stack[obj]['expectation'][2], stack[obj]['expectation'][1],
                                                   stack[obj]['expectation'][0]),
                                    gtsam.Point3(stack[obj]['expectation'][3], stack[obj]['expectation'][4],
                                                 stack[obj]['expectation'][5]))

            pose_rotation_matrix = exp_gtsam.rotation().matrix()
            cov_noise_model_rotated = self.rotate_cov_6x6(cov_noise_model, np.transpose(pose_rotation_matrix))
            cov_noise_model_rotated = gtsam.noiseModel.Gaussian.Covariance(cov_noise_model_rotated)
            #updating the graph
            self.graph.add(gtsam.PriorFactorPose3(self.XO(obj), exp_gtsam, cov_noise_model_rotated))
            graph_add.add(gtsam.PriorFactorPose3(self.XO(obj), exp_gtsam, cov_noise_model_rotated))

            self.prop_belief.add(gtsam.PriorFactorPose3(self.XO(obj), exp_gtsam, cov_noise_model_rotated))

            if self.prop_initial.exists(self.XO(obj)) is False:
                self.prop_initial.insert(self.XO(obj), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))

            # if obj in new_input_object_prior and obj not in self.classRealization:
            #     self.graph.add(gtsam.PriorFactorPose3(self.XO(obj), new_input_object_prior[obj],
            #                                           new_input_object_covariance[obj]))
            #     self.prop_belief.add(gtsam.PriorFactorPose3(self.XO(obj), new_input_object_prior[obj],
            #                                                 new_input_object_covariance[obj]))
            self.classRealization[obj] = class_realization[obj]

            if self.isam.value_exists(self.XO(obj)) is False:
                initial_add.insert(self.XO(obj), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))

            idx += 1

            if stack[obj]['weight'] > -322:
                self.logWeight += stack[obj]['weight']  # + 150
            else:
                self.logWeight = -math.inf


    def weight_update(self, da_current_step, geo_measurements, sem_measurements, number_of_samples): #[WIP]

        self.weight_memory.append(np.exp(self.logWeight))

        # Computing marginal distributions for the required variables
        path_length = len(self.daRealization)
        self.optimize(graph_to_optimize=self.prop_belief)
        self.marginals = gtsam.Marginals(self.prop_belief, self.result)

        # Add initial values if objects are not present in the current initial values (by adding multi-robot measurements)

        # Extracting expectation and covariance from last camera pose
        camera_pose_marginal_covariance = self.marginals.marginalCovariance(self.X(path_length))

        camera_pose = self.result.atPose3(self.X(path_length)).translation().vector()
        camera_pose_rotation = self.result.atPose3(self.X(path_length)).rotation().rpy()
        #camera_pose_rotation = np.flip(camera_pose_rotation, 0)
        camera_pose = np.concatenate((camera_pose_rotation, camera_pose))

        # Rotate covariance matrix
        camera_pose_marginal_covariance_rotated = self.rotate_cov_6x6(camera_pose_marginal_covariance,
                                                                      self.result.atPose3(
                                                                          self.X(path_length)).rotation().matrix())

        # Sample from camera pose
        camera_samples = np.random.multivariate_normal(camera_pose, camera_pose_marginal_covariance_rotated, number_of_samples)

        total_det = np.linalg.det(camera_pose_marginal_covariance)

        # Extracting expectation and covariance from object poses + sampling
        object_marginal_covariance = dict()
        object_pose = dict()
        object_pose_rotation = dict()
        object_samples = dict()
        object_marginal_covariance_rotated = dict()
        for selected_object in da_current_step:
            selected_object = int(selected_object)
            #self.result.print()
            #print(selected_object)
            object_marginal_covariance[self.XO(selected_object)] = \
                self.marginals.marginalCovariance(self.XO(selected_object))
            object_pose[self.XO(selected_object)] = \
                self.result.atPose3(self.XO(selected_object)).translation().vector()
            object_pose_rotation[self.XO(selected_object)] = \
                self.result.atPose3(self.XO(selected_object)).rotation().rpy()
            #object_pose_rotation[self.XO(selected_object)] = \
            #    np.flip(object_pose_rotation[self.XO(selected_object)], 0)
            object_pose[self.XO(selected_object)] = np.concatenate((object_pose_rotation[self.XO(selected_object)],
                                                                    object_pose[self.XO(selected_object)]))

            # Rotate covariance matrix
            object_marginal_covariance_rotated[self.XO(selected_object)] = self.rotate_cov_6x6(
                object_marginal_covariance[self.XO(selected_object)],
                self.result.atPose3(self.XO(selected_object)).rotation().matrix())

            # Samples
            object_samples[self.XO(selected_object)] = np.random.multivariate_normal(
                object_pose[self.XO(selected_object)],
                object_marginal_covariance_rotated[self.XO(selected_object)], number_of_samples)

        # Initializing the weight update, will use the below loop to sum all possible updates
        weight_update = 0
        # running over all samples
        for sample_index in range(number_of_samples):

            x_pose = self.vec_to_pose3(camera_samples[sample_index])
            da_probability = 1
            log_likelihood_geo = 0
            log_likelihood_sem = 0
            # running index for the next loop
            i = 0

            for selected_object in da_current_step:

                selected_object = int(selected_object)
                # compute P(C,beta|H) TODO: Add semantic measurement likelihood
                xo_pose = self.vec_to_pose3(object_samples[self.XO(selected_object)][sample_index])

                #log_likelihood_geo -= self.geoModel.log_pdf_value(geo_measurements[i],
                #                                                 x_pose, xo_pose)
                log_likelihood_sem -= self.clsModel.log_pdf_value(sem_measurements[i],
                                                                  self.classRealization[selected_object],
                                                                  x_pose, xo_pose)

                det_cov = np.linalg.det(object_marginal_covariance[self.XO(selected_object)])
                total_det *= det_cov
                # print(self.classRealization[selected_object])
                # print(log_likelihood_sem)

                i += 1

            if self.cls_enable is True:
                weight_update += da_probability * math.exp(0.5 * log_likelihood_geo) \
                                 * math.exp(0.5 * log_likelihood_sem) / number_of_samples
            else:
                weight_update += da_probability * math.exp(0.5 * log_likelihood_geo) / number_of_samples

                #weight_update *= total_det ** (-1 / 2) * (2 * np.pi) ** (-(i + 1) / 2)

        if weight_update > 10**-322:
            self.logWeight += math.log(weight_update) #+ 150
        else:
            self.logWeight = -math.inf


    def weight_update_ML(self, da_current_step, geo_measurements, sem_measurements): #[WIP]

        self.weight_memory.append(np.exp(self.logWeight))

        # Computing marginal distributions for the required variables
        path_length = len(self.daRealization)
        self.optimize(graph_to_optimize=self.prop_belief)
        self.marginals = gtsam.Marginals(self.prop_belief, self.result)

        # Add initial values if objects are not present in the current initial values (by adding multi-robot measurements)

        # Extracting expectation and covariance from last camera pose
        camera_pose_marginal_covariance = self.marginals.marginalCovariance(self.X(path_length))

        camera_pose = self.result.atPose3(self.X(path_length)).translation().vector()
        camera_pose_rotation = self.result.atPose3(self.X(path_length)).rotation().rpy()
        #camera_pose_rotation = np.flip(camera_pose_rotation, 0)
        camera_pose = np.concatenate((camera_pose_rotation, camera_pose))

        # Rotate covariance matrix
        camera_pose_marginal_covariance_rotated = self.rotate_cov_6x6(camera_pose_marginal_covariance,
                                                                      self.result.atPose3(
                                                                          self.X(path_length)).rotation().matrix())

        # Extracting expectation and covariance from object poses + sampling
        object_marginal_covariance = dict()
        object_pose = dict()
        object_pose_rotation = dict()
        object_samples = dict()
        object_marginal_covariance_rotated = dict()
        for selected_object in da_current_step:
            selected_object = int(selected_object)
            #self.result.print()
            #print(selected_object)
            object_marginal_covariance[self.XO(selected_object)] = \
                self.marginals.marginalCovariance(self.XO(selected_object))
            object_pose[self.XO(selected_object)] = \
                self.result.atPose3(self.XO(selected_object)).translation().vector()
            object_pose_rotation[self.XO(selected_object)] = \
                self.result.atPose3(self.XO(selected_object)).rotation().rpy()
            #object_pose_rotation[self.XO(selected_object)] = \
            #    np.flip(object_pose_rotation[self.XO(selected_object)], 0)
            object_pose[self.XO(selected_object)] = np.concatenate((object_pose_rotation[self.XO(selected_object)],
                                                                    object_pose[self.XO(selected_object)]))

            # Rotate covariance matrix
            object_marginal_covariance_rotated[self.XO(selected_object)] = self.rotate_cov_6x6(
                object_marginal_covariance[self.XO(selected_object)],
                self.result.atPose3(self.XO(selected_object)).rotation().matrix())

        # Initializing the weight update, will use the below loop to sum all possible updates
        weight_update = 0
        # running over all samples

        x_pose = self.vec_to_pose3(camera_pose)
        da_probability = 1
        log_likelihood_geo = 0
        log_likelihood_sem = 0
        # running index for the next loop
        i = 0

        for selected_object in da_current_step:

            selected_object = int(selected_object)
            # compute P(C,beta|H) TODO: Add semantic measurement likelihood
            xo_pose = self.vec_to_pose3(object_pose[self.XO(selected_object)])

            #log_likelihood_geo -= self.geoModel.log_pdf_value(geo_measurements[i],
            #                                                 x_pose, xo_pose)
            log_likelihood_sem -= self.clsModel.log_pdf_value(sem_measurements[i],
                                                              self.classRealization[selected_object],
                                                              x_pose, xo_pose)
            # print(self.classRealization[selected_object])
            # print(log_likelihood_sem)

            i += 1

        if self.cls_enable is True:
            weight_update += da_probability * math.exp(0.5 * log_likelihood_geo) \
                             * math.exp(0.5 * log_likelihood_sem)
        else:
            weight_update += da_probability * math.exp(0.5 * log_likelihood_geo)

        if weight_update > 10**-322:
            self.logWeight += math.log(weight_update) #+ 150
        else:
            self.logWeight = -math.inf




    # Optimize function: if graph is designated it will optimize on that graph. If not, it will optimize on self.graph
    def optimize(self, graph_to_optimize=None):

        # Optimization
        params = gtsam.LevenbergMarquardtParams()
        if graph_to_optimize is None:
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        else:
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph_to_optimize, self.prop_initial, params)

        self.result = optimizer.optimize()
        self.initial = self.result
        self.resultsFlag = True

    def optimize_isam2(self, graph_to_optimize=None): #TODO: Make ISAM2 work

        if graph_to_optimize is None:
            self.isam.update(self.graph, self.initial)
        else:
            self.isam.update(graph_to_optimize, self.initial)



        self.result = self.isam.calculate_estimate()
        self.initial = self.result
        # self.initial.print()
        self.resultsFlag = True


    def print_results(self):

        # Optimizing if not done before
        if self.resultsFlag is False:
            self.optimize()

        # Printing results
        self.graph.print("\nFactor Graph:\n-----------------\n\n")
        self.initial.print("\nInitial Estimate:\n----------------\n\n")
        self.result.print("\nFinal Result:\n----------------\n\n")

    # Convert Pose2 object to a more manageable Numpy object
    @staticmethod
    def pose2_to_list(pose2_var):

        var = list()
        var.append(pose2_var.x())
        var.append(pose2_var.y())
        var.append(pose2_var.theta())
        return var

    # Convert length 6 vector with xyz and ypr values to Pose3
    @staticmethod
    def vec_to_pose3(vector):
        pose3_t = gtsam.Point3(vector[3], vector[4], vector[5])
        pose3_r = gtsam.Rot3.Ypr(vector[2], vector[1], vector[0])
        return gtsam.Pose3(pose3_r, pose3_t)

    @staticmethod
    def rotate_cov_6x6(cov_matrix, rotation_3x3):

        rotation_6x6 = np.eye(6)
        rotation_6x6[3:6,3:6] = rotation_3x3
        rot_1 = np.matmul(rotation_6x6, cov_matrix)
        rotated_cov_matrix = np.matmul(rot_1, np.transpose(rotation_6x6))

        return rotated_cov_matrix

    # Function to display results in a figure
    def display_graph(self, fig_num=0, plt_size=10):

        # Figure ID
        fig = plt.figure(fig_num)
        ax = fig.gca()
        plt.gca()

        if self.resultsFlag is False:
            self.optimize()
        plotterdaac2d.plot2DPoints(fig_num, self.result, 'rx')

        # Plot cameras
        idx = 0
        while self.result.exists(self.X(idx)):

            pose_x = self.result.atPose3(self.X(idx))
            plotterdaac2d.plotPose2(fig_num, pose_x, 0.3)

            # Annotate the point
            ax.text(pose_x.x(), pose_x.y(), 'X' + str(idx))

            # Plot covariance ellipse
            mu, cov = plotterdaac2d.extract_mu_cov_2d(self.result, self.marginals, self.X(idx))

            # Rotate covariance matrix
            # cov_rotated = plotterdaac2d.rotate_covariance(pose_x, cov)
            angles = pose_x.rotation().matrix()
            psi = np.arctan2(angles[1, 0], angles[0, 0])
            gRp = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
            multi1 = np.matmul(gRp, cov)
            cov_rotated = np.matmul(multi1, np.transpose(gRp))

            # Plot rotated cov ellipse
            plotterdaac2d.plot_ellipse(ax, mu, cov_rotated)

            idx += 1


        # Plot objects
        idx = 1
        # self.result.print()
        while self.result.exists(self.XO(idx)):

            pose_xo = self.result.atPose3(self.XO(idx))
            plotterdaac2d.plotPoint2(fig_num, pose_xo.translation(), 'bo')

            # Annotate the point
            ax.text(pose_xo.x(), pose_xo.y(), 'O' + str(idx))

            # Plot covariance ellipse
            mu, cov = plotterdaac2d.extract_mu_cov_2d(self.result, self.marginals, self.XO(idx))

            # Rotate covariance matrix
            angles = pose_xo.rotation().matrix()
            psi = np.arctan2(angles[1, 0], angles[0, 0])
            gRp = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
            multi1 = np.matmul(gRp, cov)
            cov_rotated = np.matmul(multi1, np.transpose(gRp))

            # Plot rotated cov ellipse
            plotterdaac2d.plot_ellipse(ax, mu, cov_rotated)

            idx += 1


        plt.title('Figure number: ' + str(fig_num) + '. \nData association realization: ' + str(self.daRealization) +
                 '. \nClass realization: ' + str(self.classRealization) + '.')
        ax.set_xlabel('X axis [m]')
        ax.set_ylabel('Y axis [m]')
        # ax.set_zlabel('Z axis [m]')

        # draw
        plt.show()

    def print_covariance(self):

        keys = self.result.keys()
        for key in keys:
            cov = self.marginals.marginalCovariance(key)
            print('Covariance matrix of key ' + str(key) + ' is: ' +str(cov))

    # Plot a graph of weight to time
    def weight_graph(self, fig, ax):

        np_weight_log = np.array(self.weight_memory)
        np_weight = np.exp(np_weight_log)
        ax.plot(list(range(1, len(self.weight_memory)+1)),np_weight)

        ax.set(xlabel='Time step', ylabel='Probability of weight',
               title='Weight development of class realization ' + str(self.classRealization) +
                     ' \nand DA realization ' +str(self.daRealization))
        ax.grid()


    def record_belief(self, filename=str()):

        self.optimize()
        f = open(filename + '_object_poses', 'w')
        f2 = open(filename + '_camera_poses', 'w')

        c = open(filename + '_object_covariance', 'w')
        c2 = open(filename + '_camera_covariance', 'w')

        for i in range(len(self.classRealization)):

            # Save poses of objects
            pose_translation = self.result.atPose3(self.XO(i + 1)).translation().vector()
            pose_rotation = self.result.atPose3(self.XO(i + 1)).rotation().rpy()
            f.write(str(pose_translation[0]) + '\t' + str(pose_translation[1])
                    + '\t' + str(pose_translation[2]) + '\t' + str(pose_rotation[2])
                    + '\t' + str(pose_rotation[1]) + '\t' + str(pose_rotation[0]) + '\n')
            # Save covariance matrices of objects
            cov_matrix = self.marginals.marginalCovariance(self.XO(i+1))
            c.write(str(cov_matrix.flatten()) + '\n')

        for i in range(len(self.daRealization) + 1):
            # Save poses of objects
            pose_translation = self.result.atPose3(self.X(i)).translation().vector()
            pose_rotation = self.result.atPose3(self.X(i)).rotation().rpy()
            f2.write(str(pose_translation[0]) + '\t' + str(pose_translation[1])
                    + '\t' + str(pose_translation[2]) + '\t' + str(pose_rotation[2])
                    + '\t' + str(pose_rotation[1]) + '\t' + str(pose_rotation[0]) + '\n')
            # Save covariance matrices of objects
            cov_matrix = self.marginals.marginalCovariance(self.X(i))
            c2.write(str(cov_matrix.flatten()) + '\n')

        f.close()
        f2.close()

        c.close()
        c2.close()