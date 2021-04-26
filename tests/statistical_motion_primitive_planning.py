from __future__ import print_function
import numpy as np
import scipy as sp
from scipy import special
from scipy import stats
import gtsam
import math
import damodel
import geomodel as geomodel_proj
import plotterdaac2d
import gaussianb_jlp
import matplotlib.pyplot as plt
from JLP_planner import JLPPLannerPrimitives
from hybridb import HybridBelief
from lambdab_lg import LambdaBelief
from lambda_planner import LambdaPlannerPrimitives
import time
import clsmodel_lg1 as clsmodel_lg1
import clsUmodel_fake_1d as clsUmodel_fake_1d
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

def motion_primitive_planning(seed, Lambda_BSP_mode, reward_mode, cls_model, number_of_beliefs,
                              random_objects=True, use_bounds = True, plt_ax = None, bar_ax = None):

    # Initializations ----------------------------------------------------
    sample_use_flag = False
    ML_planning_flag = True
    track_length = 20
    horizon = 8
    # if number_of_beliefs == 1:
    #     horizon = 4
    ML_update = True
    MCTS_flag = True
    MCTS_branches = 3
    cls_enable = True
    display_graphs = True
    measurement_radius_limit = 10
    measurement_radius_minimum = 0
    com_radius = 100000
    if random_objects is False:
        class_GT = {1: 2, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2}#, 10: 1}
    else:
        class_GT = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1}#, 6: 1, 7: 1, 8: 1, 9: 2}  # , 10: 1}
    np.random.seed(seed)
    measurements_enabler = True
    opening_angle = 60
    opening_angle_rad = opening_angle * math.pi / 180
    num_samp = 50
    CVaR_flag = True
    start_time = time.time()
    entropy_lower_limit = -5

    # Set models ---------------------------------------------------------

    np.set_printoptions(precision=3)
    plt.rcParams.update({'font.size': 16})

    prior_lambda = (1, 1)
    lambda_prior = np.array([special.digamma(prior_lambda[0]) - special.digamma(prior_lambda[1])]) #
    lambda_prior_noise_diag = np.matrix(special.polygamma(1, prior_lambda[0]) + special.polygamma(1, prior_lambda[1]))
    lambda_prior_noise = gtsam.noiseModel.Diagonal.Variances(lambda_prior_noise_diag)

    da_model = damodel.DAModel(camera_fov_angle_horizontal=opening_angle_rad, range_limit=measurement_radius_limit,
                               range_minimum=measurement_radius_minimum)
    geo_model_noise_diag = np.array([0.00001, 0.00001, 0.001, 0.0001, 0.0001, 0.001])
    geo_model_noise = gtsam.noiseModel.Diagonal.Variances(geo_model_noise_diag)
    geo_model = geomodel_proj.GeoModel(geo_model_noise)
    # DEFINE GROUND TRUTH -------------------------------------------------

    # Objects
    GT_objects = list()
    # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, -2.0, 0.0])))
    # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi/2, 0.0, 0.0), np.array([1.0, 2.0, 0.0])))
    # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi/2, 0.0, 0.0), np.array([-2.5, 3.5, 0.0])))
    # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3*math.pi/4, 0.0, 0.0), np.array([1.0, 4.0, 0.0])))
    # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([-6.0, -3.0, 0.0])))
    # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi/4, 0.0, 0.0), np.array([3.0, -2.0, 0.0])))
    # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3*math.pi/2, 0.0, 0.0), np.array([1.5,-6.0, 0.0])))
    # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi, 0.0, 0.0), np.array([-1.0, -4.0, 0.0])))
    # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi, 0.0, 0.0 / 2), np.array([-1.5, -1.0, 0.0])))
    # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(3.141, 0.0, 0.0), np.array([-6.5, 1.0, 0.0])))

    if random_objects is False:

        # For single object
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(3 * 3.141 / 2, 0.0, 0.0), np.array([10.5, 0.0, 0.0])))

        # For multiple objects
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([2.5, -4.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(3.141 / 2, 0.0, 0.0), np.array([4, -7.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141 / 2, 0.0, 0.0), np.array([4.5, -1.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(3.141 / 4, 0.0, 0.0), np.array([6.5, 3.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0*3.141, 0.0, 0.0), np.array([7.5, 4.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(1 * 3.141 / 2, 0.0, 0.0), np.array([8, 1.5, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141 / 2, 0.0, 0.0), np.array([8.5, 7.7, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141 / 4, 0.0, 0.0), np.array([11.5, 1.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0 * 3.141 / 4, 0.0, 0.0), np.array([12, -2.5, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-2 * 3.141 / 4, 0.0, 0.0), np.array([14.5, 2.7, 0.0])))

    elif random_objects is True:
        for idx in range(len(class_GT)):
            GT_objects.append(gtsam.Pose3(gtsam.Pose2(np.random.uniform(0, 10, 1),
                                                      np.random.uniform(-8, 8, 1),
                                                      np.random.uniform(-3.141, 3.141, 1))))

    # Init poses
    Init_poses = list()
    Init_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([-8.0, 0.2, 0.0])))
    Init_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([-8.0, 0.1, 0.0])))
    Init_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([-8.0, -0.1, 0.0])))
    Init_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([-8.0, -0.3, 0.0])))
    Init_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([-8.0, -0.4, 0.0])))
    Init_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([-8.0, -0.5, 0.0])))


    # Robot poses
    GT_poses = list()
    GT_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0*math.pi/2, 0.0, 0.0), np.array([-4.0, 0.0, 0.0])))

    # DEFINE ACTION PRIMITIVES ----------------------------------------------

    stride_length = 1.0
    motion_primitives = list()
    motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0.00, 0.0, 0.0), np.array([stride_length, 0.0, 0.0])))
    motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(+math.pi/4, 0.0, 0.0), np.array([stride_length, stride_length, 0.0])))
    motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi/4, 0.0, 0.0), np.array([stride_length, -stride_length, 0.0])))
    motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(+math.pi/2, 0.0, 0.0), np.array([0.0, stride_length, 0.0])))
    motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi/2, 0.0, 0.0), np.array([0.0, -stride_length, 0.0])))

    #------------------------------------------------ PRIORS

    # Robot priors
    #prior = gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0]))
    prior = GT_poses[0]
    prior_noise_diag = np.array([00.000001, 0.000001, 0.0000038, 00.00002, 00.0000202, 0.000001])
    prior_noise = gtsam.noiseModel.Diagonal.Variances(prior_noise_diag)
    prior_noise_cov = np.diag(prior_noise_diag)

    # Create belief based on the switch
    if Lambda_BSP_mode == 1:

        belief_list = list()
        lambda_planner_list = list()

        for idx in range(number_of_beliefs):
            # cls_prior_rand_0 = np.random.dirichlet(prior_lambda)
            cls_prior_rand_0_lg = np.random.multivariate_normal(lambda_prior, lambda_prior_noise_diag)
            cls_prior_rand_0_con = np.exp(cls_prior_rand_0_lg) / (1 + np.sum(np.exp(cls_prior_rand_0_lg)))
            cls_prior_rand_0 = np.concatenate((cls_prior_rand_0_con, 1 /
                                               (1 + np.sum(np.exp(cls_prior_rand_0_lg)))), axis=None)
            cls_prior_rand = cls_prior_rand_0.tolist()


            belief_list.append(
                HybridBelief(2, geo_model, da_model, cls_model, cls_prior_rand, GT_poses[0],
                             prior_noise,
                             cls_enable=cls_enable, pruning_threshold=8.5, ML_update=ML_update))

        lambda_belief = LambdaBelief(belief_list)
        lambda_planner = LambdaPlannerPrimitives(lambda_belief, entropy_lower_limit=entropy_lower_limit)

    if Lambda_BSP_mode == 2:

        lambda_belief = gaussianb_jlp.JLPBelief(geo_model, da_model, cls_model, GT_poses[0], prior_noise,
                                                              lambda_prior_mean=lambda_prior,
                                                              lambda_prior_noise=lambda_prior_noise_diag,
                                                              cls_enable=cls_enable)
        lambda_planner = JLPPLannerPrimitives(lambda_belief, entropy_lower_limit=entropy_lower_limit)

    # Plot all the GT sequences
    # fig = plt.figure(0)
    # ax = fig.gca()
    # plotterdaac2d.GT_plotter_line(GT_poses, GT_objects, color=(0.8, 0.4, 0.4),
    #                               limitx=[-10, 7], limity=[-8, 8], show_plot=False, ax=ax,
    #                               start_pose=GT_poses[0])
    # ax.set_xlabel('X axis [m]')
    # ax.set_ylabel('Y axis [m]')
    # plt.tight_layout()
    # plt.show()

    # SEMANTIC MEASUREMENT GENERATION FUNCTION-----------------------------------------------------

    def sem_measurements_generator(relative_pose, no_noise=False):

        radius = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2)
        psi = np.arctan2(relative_pose.y(), relative_pose.x())
        theta = np.arctan2(-relative_pose.z(), radius)

        K = 2
        Alpha = 0.5

        R = K * (0.7 + 0.3 * np.cos(psi + theta))
        Sigma = 1 / R ** 2

        f = Alpha * np.cos(2 * psi + 2 * theta) + Alpha

        sem_gen = -1

        if no_noise is False:
            sem_gen = np.random.normal(f, Sigma)
        else:
            sem_gen = f

        sem_gen_vector = [ np.exp(sem_gen) / (1 + np.exp(sem_gen)), 1 / (1 + np.exp(sem_gen))]

        return sem_gen_vector


    def sem_measurements_generator_2(relative_pose, no_noise=False):

        radius = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2)
        psi = np.arctan2(relative_pose.y(), relative_pose.x())
        theta = np.arctan2(-relative_pose.z(), radius)

        K = 2
        Alpha = -0.5

        R = K * (0.7 + 0.3 * np.cos(psi + theta))
        Sigma = 1 / R ** 2

        f = Alpha * np.cos(2 * psi + 2 * theta) + Alpha

        sem_gen = -1

        if no_noise is False:
            sem_gen = np.random.normal(f, Sigma)
        else:
            sem_gen = f

        sem_gen_vector = [ np.exp(sem_gen) / (1 + np.exp(sem_gen)), 1 / (1 + np.exp(sem_gen))]

        return sem_gen_vector


    # GEOMETRIC MEASUREMENT GENERATOR -------------------------------------------------------------------

    def geo_measurement_generator(relative_pose, meas_cov, no_noise=False):

        # Break down pose to components
        pose_pos = [relative_pose.x(), relative_pose.y(), relative_pose.z()]
        pose_rot = relative_pose.rotation().rpy()

        #
        pose_total = np.concatenate((pose_rot, pose_pos))

        # Create the pose
        if no_noise is False:
            pose_sample = np.random.multivariate_normal(pose_total, meas_cov)
        else:
            pose_sample = pose_total

        # Transfer to pose3
        pose_generated = gtsam.Pose3(gtsam.Rot3.Ypr(pose_sample[2], pose_sample[1], pose_sample[0]),
                                         np.array([pose_sample[3], pose_sample[4], pose_sample[5]]))

        return pose_generated

    # ACTION GENERATOR

    def action_generator(action, action_noise_diag, no_noise=False):
        """

        :type action: gtsam.Pose3
        """
        action_noise = np.diag(action_noise_diag)
        action_rotation = action.rotation().rpy()
        action_vector = np.array([action_rotation[2], action_rotation[1], action_rotation[0],
                                  action.x(), action.y(), action.z()])
        generated_action = np.random.multivariate_normal(action_vector, action_noise)
        generated_action_pose = gtsam.Pose3(gtsam.Rot3.Ypr(generated_action[0], generated_action[1], generated_action[2]),
                                            np.array([generated_action[3], generated_action[4], generated_action[5]]))
        if no_noise is False:
            return generated_action_pose
        else:
            return action

    def lambda_entropy_individual_numeric(exp, covariance, number_of_samples=100):

        entropy = 0
        for idx in range(number_of_samples):

            sample = np.random.multivariate_normal(exp, covariance)
            sample_aug = np.concatenate((sample, 0), axis=None)
            sample_cpv = np.exp(sample_aug)/(np.sum(np.exp(sample_aug)))
            log_pdf_value = np.log(stats.multivariate_normal.pdf(sample, exp, covariance)) - \
                        np.sum(np.log(sample_cpv))
            entropy -= log_pdf_value / number_of_samples

        return entropy


    prior_entropy = lambda_entropy_individual_numeric(lambda_prior, lambda_prior_noise_diag)

    # INFERENCE
    action_noise_diag = np.array([0.0003, 0.0003, 0.0001, 0.0003, 0.00030, 0.001])
    action_noise = gtsam.noiseModel.Diagonal.Variances(action_noise_diag)

    reward_collections = list()
    chosen_action_list = list()
    current_entropy_list = list()
    current_entropy_list.append(0)
    current_R2_list = list()
    current_R2_list.append(np.NaN)
    sigma_list = list()
    sigma_list.append(0)
    time_list = list()
    time_list.append(0)
    MSDE = list()
    MSDE.append(1/4)

    #for sequence_idx in range(track_length):
    for idx in range(track_length):

        step_time = time.time()

        action, reward = lambda_planner.planning_session(action_noise, np.diag(geo_model_noise_diag),
                                                         horizon=horizon, reward_print=True,
                                                         enable_MCTS=MCTS_flag, MCTS_braches_per_action=MCTS_branches,
                                                         ML_planning=ML_planning_flag, sample_use_flag=sample_use_flag,
                                                         reward_mode=reward_mode, motion_primitives=motion_primitives,
                                                         use_lower_bound=use_bounds)

        chosen_action_list.append(action)


        # action = motion_primitives[0]
        print('----------------------------------------------------------------------------')
        print('Action:\n' + str(action))
        # print('Reward: ' + str(reward))
        GT_poses.append(GT_poses[idx].compose(action))
        action_generated = action_generator(action, action_noise_diag, no_noise=True)
        print('idx ' + str(idx))
        lambda_belief.action_step(action_generated, action_noise) #TODO: see if belief within the planner is update

        # Defining opening angle
        geo_measurements = list()
        sem_measurements = list()
        for lambda_real in range(num_samp):
            sem_measurements.append(list())
        GT_DA = list()

        # Generate measurements and forward the beliefs
        for j in range(len(GT_objects)):

            # Check if the object is seen
            xy_angle = np.arctan2(GT_objects[j].y() - GT_poses[idx + 1].y(),
                                  GT_objects[j].x() - GT_poses[idx + 1].x())
            angles = GT_poses[idx + 1].rotation().matrix()
            psi = np.arctan2(angles[1, 0], angles[0, 0])
            geo_part = GT_poses[idx + 1].between(GT_objects[j])
            geo_radius = np.sqrt(geo_part.x() ** 2 + geo_part.y() ** 2 + geo_part.z() ** 2)

            # if np.abs(xy_angle - psi) <= opening_angle_rad and geo_radius <= measurement_radius_limit and \
            #         geo_radius >= measurement_radius_minimum:

            if da_model.object_observation_3d(GT_poses[idx + 1], GT_objects[j]) is 1:

                GT_DA.append(j + 1)

                geo_cov = np.diag(geo_model_noise_diag)
                meas_relative_pose = geo_measurement_generator(geo_part, geo_cov, no_noise=False)

                geo_measurements.append(meas_relative_pose)
                # print(geo_measurements)

                for lambda_real in range(num_samp):

                    if class_GT[j + 1] == 2:
                        sem_part = sem_measurements_generator_2(GT_objects[j].between(GT_poses[idx + 1]),
                                                                no_noise=False)
                        #print('Object: ' + str(j + 1) + ',sem: ' + str(sem_part) + ',robot ' + robot_id + ',Class 2')
                    else:
                        sem_part = sem_measurements_generator(GT_objects[j].between(GT_poses[idx + 1]),
                                                              no_noise=False)

                    sem_measurements[lambda_real].append(sem_part)  # GENERATED




        # -------------------------------###
        if Lambda_BSP_mode == 1:
            lambda_belief.add_measurements(geo_measurements, sem_measurements, da_current_step=GT_DA,
                                                              number_of_samples=num_samp, new_input_object_prior=None,
                                                              new_input_object_covariance=None)

        # -------------------------------###
        if Lambda_BSP_mode == 2:

            sem_measurements_exp = list()
            sem_measurements_cov = list()

            obj_idx_running = 0
            # Go over all objects
            for obj_idx in GT_DA:

                obj_data = np.zeros((num_samp, len(prior_lambda)))
                for b_idx in range(num_samp):
                    obj_data[b_idx, :] = sem_measurements[b_idx][obj_idx_running]
                #obj_data = np.array(sem_measurements[obj_idx - 1])
                obj_data_logit = np.array(np.log(obj_data[:, 0] / obj_data[:, 1]))

                sem_measurements_exp.append([np.sum(obj_data_logit, axis=0) / num_samp])
                sem_measurements_cov.append(np.matrix(np.dot(np.transpose(obj_data_logit - sem_measurements_exp[-1]),
                                                      obj_data_logit - sem_measurements_exp[-1]) / (num_samp - 1)))
                obj_idx_running += 1

            lambda_belief.add_measurements(geo_measurements, sem_measurements_exp, sem_measurements_cov,
                                                              GT_DA)

        lambda_belief.print_lambda_lg_params()

        entropy_collector = 0
        sigma_collector = 0
        R2_collector = 0
        entropy_lower_limit = -12.13
        if Lambda_BSP_mode == 1:

            for belief in lambda_belief.belief_list:
                object_list = belief.obj_realization
                break

            for obj in object_list:
                entropy_collector += lambda_belief.entropy(obj, entropy_lower_limit=entropy_lower_limit)
                sigma_collector += 2 * np.sqrt(lambda_belief.lambda_covariance(obj)[0, 0]) / len(object_list)
                R2_collector += lambda_belief.entropy_lambda_expectation(obj)

            if not object_list:
                R2_collector = np.NaN

        if Lambda_BSP_mode == 2:

            for obj in lambda_belief.object_lambda_dict:
                entropy_collector += lambda_belief.\
                    lambda_entropy_individual_numeric(lambda_belief.object_lambda_dict[obj],
                                                      entropy_lower_limit=entropy_lower_limit)
                sigma_collector += 2 * np.sqrt(lambda_belief.
                                               lambda_covariance_numeric(
                    lambda_belief.object_lambda_dict[obj])[0, 0]) / len(lambda_belief.object_lambda_dict)
                R2_collector += lambda_belief.lambda_expectation_entropy(lambda_belief.object_lambda_dict[obj])

            if not lambda_belief.object_lambda_dict:
                R2_collector = np.NaN

        current_entropy_list.append(entropy_collector)
        MSDE.append(lambda_belief.MSDE_expectation(class_GT))
        sigma_list.append(sigma_collector)
        current_R2_list.append(R2_collector)

        time_list.append(time.time() - step_time)

    print(chosen_action_list)

    # if Lambda_BSP_mode == 1:
    #     ax = lambda_belief.belief_list[0].graph_realizations_in_one(idx + 1, fig_num=0, show_obj=True,
    #                                                                 show_weights=False, show_plot=False,
    #                                                                 show_elipses=False)
    # if Lambda_BSP_mode == 2:
    #     ax = lambda_belief.display_graph(plot_line=True, display_title=False, show_plot=False)

    width_offset = -0.36
    if reward_mode == 1:
        width_offset = -0.36
        color_bar = 'purple'
    elif reward_mode == 2:
        width_offset = 0.0
        color_bar = 'red'
    if Lambda_BSP_mode == 2:
        if reward_mode == 1:
            width_offset = -0.18
            color_bar = 'black'
        elif reward_mode == 2:
            width_offset = 0.18
            color_bar = 'blue'
    elif Lambda_BSP_mode == 1 and number_of_beliefs == 1:
        width_offset = 0.36
        color_bar = 'green'

    # if reward_mode == 1:
    #     color = 'r'
    # elif reward_mode == 2 and number_of_beliefs == 1:
    #     color = 'g'
    # elif reward_mode == 2 and number_of_beliefs != 1:
    #     color = 'b'

    plotterdaac2d.GT_plotter_line(GT_poses, GT_objects, fig_num=0, show_plot=False,
                                  plt_save_name=None, pause_time=None, limitx=[-2, 15], limity=[-8, 8],
                                  jpg_save=False, alpha=0.75, color=color_bar, ax=plt_ax)

    # Add cones
    if Lambda_BSP_mode == 1:
        wedges = []
        obj_idx_wedge = 0
        for GT_object in GT_objects:
            obj_idx_wedge += 1
            if class_GT[obj_idx_wedge] == 1:
                wedge = mpatches.Wedge((GT_object.x(), GT_object.y()), measurement_radius_limit,
                                       GT_object.rotation().yaw() * 180 / 3.141 - 15,
                                       GT_object.rotation().yaw() * 180 / 3.141 + 15, ec="none")
            else:
                wedge = mpatches.Wedge((GT_object.x(), GT_object.y()), measurement_radius_limit,
                                       GT_object.rotation().yaw() * 180 / 3.141 - 15,
                                       GT_object.rotation().yaw() * 180 / 3.141 + 15, ec="none")
            wedges.append(wedge)
        collection = PatchCollection(wedges, facecolors='blue', alpha=0.2)
        plt_ax.add_collection(collection)

    plt_ax.set_xlim(-8, 15)
    plt_ax.set_ylim(-10, 10)



    lambda_belief.lambda_bar_error_graph(show_plt=False, ax=bar_ax, color=color_bar, width_offset=width_offset,
                                         GT_classes=class_GT, GT_bar=True)
    plt.tight_layout()

    return current_entropy_list, MSDE, sigma_list, time.time() - start_time, time_list, current_R2_list

# Compute standard deviation
def compute_sigma_vec(list_2d, expectation_vector):

    list_2d_np = np.array(list_2d)
    shape = np.shape(list_2d_np)

    expectation_vector_expanded = np.tile(expectation_vector, (shape[0], 1))

    var_vec = np.matmul(np.transpose(list_2d_np - expectation_vector_expanded),
                                  list_2d_np - expectation_vector_expanded) / (len(expectation_vector) - 1)
    var_vec_diag = np.diagonal(var_vec)
    sigma_vec_diag = np.sqrt(var_vec_diag)
    return sigma_vec_diag



#-----------------------------------------------------------------------------
# Main function; runs the statistical study.
def main():

    plt.rcParams.update({'font.size': 18})

    activate_MH = True
    number_of_seeds = 1
    number_of_beliefs = 5
    seed_offset = 102
    cls_model_1 = clsmodel_lg1.ClsModel()
    cls_model_2 = clsUmodel_fake_1d.JLPModel()
    random_object_flag = False

    fig_1, ax_red = plt.subplots()
    fig_2, ax_blue = plt.subplots()
    fig_3, ax_green = plt.subplots()

    fig_4, ax_bar_r1 = plt.subplots()

    if number_of_seeds == 1:
        fixed_stat = 'Fixed'
    else:
        fixed_stat = 'Stat'

    JLP_entropy_R1 = list()

    JLP_MSDE_R1 = list()

    JLP_sigma_R1 = list()

    JLP_e_entropy_R1 = list()

    JLP_entropy_R2 = list()

    JLP_MSDE_R2 = list()

    JLP_sigma_R2 = list()

    JLP_e_entropy_R2 = list()

    JLP_entropy_R3 = list()

    JLP_MSDE_R3 = list()

    JLP_sigma_R3 = list()

    MH_entropy_R1 = list()

    MH_e_entropy_R1 = list()

    MH_MSDE_R1 = list()

    MH_sigma_R1 = list()

    MH_entropy_R2 = list()

    MH_e_entropy_R2 = list()

    MH_MSDE_R2 = list()

    MH_sigma_R2 = list()

    MH_entropy_R3 = list()

    MH_MSDE_R3 = list()

    MH_sigma_R3 = list()

    WEU_entropy = list()

    WEU_MSDE = list()

    time_dict = {'JLP_R1': list(), 'MH_R1': list(), 'JLP_R2': list(),'MH_R2': list(), 'WEU': list()}

    time_dict_track = {'JLP_R1': None, 'MH_R1': None, 'JLP_R2': None,'MH_R2': None, 'WEU': None}


    for seed_idx in range(number_of_seeds):

        # Patten18arj like planning
        Lambda_BSP_mode = 1
        reward_mode = 2
        entropy, MSDE, _, time, time_dict_track['WEU'], _ = \
            motion_primitive_planning(seed_idx + seed_offset + 4, Lambda_BSP_mode, reward_mode, cls_model_1,
                                      number_of_beliefs=1, random_objects=random_object_flag,
                                      plt_ax=ax_green, bar_ax=ax_bar_r1)#TODO: switch to r2 when needed.
        WEU_entropy.append(entropy)
        WEU_MSDE.append(MSDE)

        time_dict['WEU'].append(time)

        if seed_idx == 0:
            WEU_MSDE_exp = np.array(MSDE) / number_of_seeds
        else:
            WEU_MSDE_exp += np.array(MSDE) / number_of_seeds

        # JLP planning lambda entropy
        Lambda_BSP_mode = 2
        reward_mode = 1

        entropy, MSDE, sigma, time, time_dict_track['JLP_R1'], eentropy = \
            motion_primitive_planning(seed_idx + seed_offset + 9, Lambda_BSP_mode, reward_mode, cls_model_2,
                                      number_of_beliefs=200, random_objects=random_object_flag,
                                      plt_ax=ax_red, bar_ax=ax_bar_r1)
        JLP_entropy_R1.append(entropy)
        JLP_MSDE_R1.append(MSDE)
        JLP_sigma_R1.append(sigma)
        JLP_e_entropy_R1.append(eentropy)

        time_dict['JLP_R1'].append(time)

        if seed_idx == 0:
            JLP_entropy_exp_R1 = np.array(entropy) / number_of_seeds
            JLP_MSDE_exp_R1 = np.array(MSDE) / number_of_seeds
            JLP_sigma_exp_R1 = np.array(sigma) / number_of_seeds
            JLP_e_entropy_exp_R1 = np.array(eentropy) / number_of_seeds
        else:
            JLP_entropy_exp_R1 += np.array(entropy) / number_of_seeds
            JLP_MSDE_exp_R1 += np.array(MSDE) / number_of_seeds
            JLP_sigma_exp_R1 += np.array(sigma) / number_of_seeds
            JLP_e_entropy_exp_R1 += np.array(eentropy) / number_of_seeds

        # JLP planning lambda expectation entropy
        Lambda_BSP_mode = 2
        reward_mode = 2

        entropy, MSDE, sigma, time, time_dict_track['JLP_R2'], eentropy = \
            motion_primitive_planning(seed_idx + seed_offset + 132, Lambda_BSP_mode, reward_mode, cls_model_2,
                                      number_of_beliefs=200, random_objects=random_object_flag,
                                      plt_ax=ax_blue, bar_ax=ax_bar_r1)
        JLP_entropy_R2.append(entropy)
        JLP_MSDE_R2.append(MSDE)
        JLP_sigma_R2.append(sigma)
        JLP_e_entropy_R2.append(eentropy)

        time_dict['JLP_R2'].append(time)

        if seed_idx == 0:
            JLP_entropy_exp_R2 = np.array(entropy) / number_of_seeds
            JLP_MSDE_exp_R2 = np.array(MSDE) / number_of_seeds
            JLP_sigma_exp_R2 = np.array(sigma) / number_of_seeds
            JLP_e_entropy_exp_R2 = np.array(eentropy) / number_of_seeds
        else:
            JLP_entropy_exp_R2 += np.array(entropy) / number_of_seeds
            JLP_MSDE_exp_R2 += np.array(MSDE) / number_of_seeds
            JLP_sigma_exp_R2 += np.array(sigma) / number_of_seeds
            JLP_e_entropy_exp_R2 += np.array(eentropy) / number_of_seeds

        if activate_MH is True:
            # MH planning lambda entropy
            Lambda_BSP_mode = 1
            reward_mode = 1

            entropy, MSDE, sigma, time, time_dict_track['MH_R1'], eentropy = \
                motion_primitive_planning(seed_idx + seed_offset + 26, Lambda_BSP_mode, reward_mode, cls_model_1,
                                          number_of_beliefs=number_of_beliefs, random_objects=random_object_flag,
                                          plt_ax=ax_red, bar_ax=ax_bar_r1)
            MH_entropy_R1.append(entropy)
            MH_MSDE_R1.append(MSDE)
            MH_sigma_R1.append(sigma)
            MH_e_entropy_R1.append(eentropy)

            time_dict['MH_R1'].append(time)

            if seed_idx == 0:
                MH_entropy_exp_R1 = np.array(entropy) / number_of_seeds
                MH_MSDE_exp_R1 = np.array(MSDE) / number_of_seeds
                MH_sigma_exp_R1 = np.array(sigma) / number_of_seeds
                MH_e_entropy_exp_R1 = np.array(eentropy) / number_of_seeds
            else:
                MH_entropy_exp_R1 += np.array(entropy) / number_of_seeds
                MH_MSDE_exp_R1 += np.array(MSDE) / number_of_seeds
                MH_sigma_exp_R1 += np.array(sigma) / number_of_seeds
                MH_e_entropy_exp_R1 += np.array(eentropy) / number_of_seeds

            # MH planning lambda expectation entropy
            Lambda_BSP_mode = 1
            reward_mode = 2

            entropy, MSDE, sigma, time, time_dict_track['MH_R2'], eentropy = \
                motion_primitive_planning(seed_idx + seed_offset + 40, Lambda_BSP_mode, reward_mode, cls_model_1,
                                          number_of_beliefs=number_of_beliefs, random_objects=random_object_flag,
                                          plt_ax=ax_blue, bar_ax=ax_bar_r1)
            MH_entropy_R2.append(entropy)
            MH_MSDE_R2.append(MSDE)
            MH_sigma_R2.append(sigma)
            MH_e_entropy_R2.append(eentropy)

            time_dict['MH_R2'].append(time)

            if seed_idx == 0:
                MH_entropy_exp_R2 = np.array(entropy) / number_of_seeds
                MH_MSDE_exp_R2 = np.array(MSDE) / number_of_seeds
                MH_sigma_exp_R2 = np.array(sigma) / number_of_seeds
                MH_e_entropy_exp_R2 = np.array(eentropy) / number_of_seeds
            else:
                MH_entropy_exp_R2 += np.array(entropy) / number_of_seeds
                MH_MSDE_exp_R2 += np.array(MSDE) / number_of_seeds
                MH_sigma_exp_R2 += np.array(sigma) / number_of_seeds
                MH_e_entropy_exp_R2 += np.array(eentropy) / number_of_seeds

        print('-O-O---------------------------SEED: ' +str(seed_idx) + ' --------------------------O-O-')




    ax_red.set_xlabel('X axis [m]')
    ax_red.set_ylabel('Y axis [m]')
    # ax_red.set_xlim(-10, 10)
    # ax_red.set_ylim(-10, 10)
    ax_blue.set_xlabel('X axis [m]')
    ax_blue.set_ylabel('Y axis [m]')
    # ax_blue.set_xlim(-10, 10)
    # ax_blue.set_ylim(-10, 10)
    ax_green.set_xlabel('X axis [m]')
    ax_green.set_ylabel('Y axis [m]')
    # ax_green.set_xlim(-10, 10)
    # ax_green.set_ylim(-10, 10)
    #ax_bar_r2.legend(['JLP R2', 'MH R2'])
    ax_bar_r1.legend(['WEU', 'JLP R1', 'JLP R2',  'MH R1', 'MH R2'], bbox_to_anchor=(0.5, 1), loc='lower center',
                     ncol=5)

    fig_1.tight_layout()
    fig_2.tight_layout()
    fig_3.tight_layout()
    fig_4.tight_layout()

    # ax_red.tight_layout()
    # ax_blue.tight_layout()
    # ax_green.tight_layout()
    # ax_bar_r1.tight_layout()
    # ax_bar_r2.tight_layout()


    plt.show()

    # R1 comparison MH
    if activate_MH is True:
        fig = plt.figure(100)
        for idx in range(number_of_seeds):
            # plt.plot(JLP_entropy_R1[idx], color='black', alpha=1 / number_of_seeds, label='JLP')
            plt.plot(MH_entropy_R1[idx], color='purple', alpha=1 / number_of_seeds, label='MH R1')
            # plt.plot(JLP_entropy_R2[idx], color='blue', alpha=1 / number_of_seeds, label='JLP')
            plt.plot(MH_entropy_R2[idx], color='red', alpha=1 / number_of_seeds, label='MH R2')
        #plt.title(r'Entropy of $b[\lambda]$ over time')
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        plt.legend(loc='lower left')
        fig.savefig('../figures/Plan_Fixed_MH_R1.eps')
        plt.show()

    # R2 comparison MH
    if activate_MH is True:
        fig = plt.figure(108)
        for idx in range(number_of_seeds):
            # plt.plot(JLP_entropy_R1[idx], color='black', alpha=1 / number_of_seeds, label='JLP')
            plt.plot(MH_e_entropy_R1[idx], color='purple', alpha=1 / number_of_seeds, label='MH R1')
            # plt.plot(JLP_entropy_R2[idx], color='blue', alpha=1 / number_of_seeds, label='JLP')
            plt.plot(MH_e_entropy_R2[idx], color='red', alpha=1 / number_of_seeds, label='MH R2')
        #plt.title(r'Entropy of $b[\lambda]$ over time')
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        plt.legend(loc='lower left')
        fig.savefig('../figures/Plan_Fixed_MH_R2.eps')
        plt.show()

    # R1 comparison JLP
    fig = plt.figure(101)
    for idx in range(number_of_seeds):
        plt.plot(JLP_entropy_R1[idx], color='black', alpha=1 / number_of_seeds, label='JLP R1')
        # plt.plot(-MH_entropy_R1[idx], color='purple', alpha=1 / number_of_seeds, label='MH R1')
        plt.plot(JLP_entropy_R2[idx], color='blue', alpha=1 / number_of_seeds, label='JLP R2')
        # plt.plot(-MH_entropy_R2[idx], color='red', alpha=1 / number_of_seeds, label='MH R2')
    #plt.title(r'Entropy of $b[\lambda]$ over time')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.tight_layout()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.legend(loc='lower left')
    fig.savefig('../figures/Plan_Fixed_JLP_R1.eps')
    plt.show()

    # R2 comparison JLP
    fig = plt.figure(109)
    for idx in range(number_of_seeds):
        plt.plot(JLP_e_entropy_R1[idx], color='black', alpha=1 / number_of_seeds, label='JLP R1')
        # plt.plot(-MH_entropy_R1[idx], color='purple', alpha=1 / number_of_seeds, label='MH R1')
        plt.plot(JLP_e_entropy_R2[idx], color='blue', alpha=1 / number_of_seeds, label='JLP R2')
        # plt.plot(-MH_entropy_R2[idx], color='red', alpha=1 / number_of_seeds, label='MH R2')
    #plt.title(r'Entropy of $b[\lambda]$ over time')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.tight_layout()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.legend(loc='lower left')
    fig.savefig('../figures/Plan_Fixed_JLP_R2.eps')
    plt.show()

    JLP_entropy_sigma_R1 = compute_sigma_vec(JLP_entropy_R1, JLP_entropy_exp_R1)
    JLP_entropy_sigma_R2 = compute_sigma_vec(JLP_entropy_R2, JLP_entropy_exp_R2)
    JLP_e_entropy_sigma_R1 = compute_sigma_vec(JLP_e_entropy_R1, JLP_e_entropy_exp_R1)
    JLP_e_entropy_sigma_R2 = compute_sigma_vec(JLP_e_entropy_R2, JLP_e_entropy_exp_R2)

    if activate_MH is True:
        MH_entropy_sigma_R1 = compute_sigma_vec(MH_entropy_R1, MH_entropy_exp_R1)
        MH_entropy_sigma_R2 = compute_sigma_vec(MH_entropy_R2, MH_entropy_exp_R2)
        MH_e_entropy_sigma_R1 = compute_sigma_vec(MH_e_entropy_R1, MH_e_entropy_exp_R1)
        MH_e_entropy_sigma_R2 = compute_sigma_vec(MH_e_entropy_R2, MH_e_entropy_exp_R2)

    # Statistical R1 comparison JLP
    fig = plt.figure(1001)
    plt.plot(JLP_entropy_exp_R1, color='black', label='JLP R1')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R1 - JLP_entropy_sigma_R1, JLP_entropy_exp_R1 +
                     JLP_entropy_sigma_R1, color='black', alpha=0.2)
    # plt.plot(MH_entropy_exp_R1, color='purple', label='MH R1')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R1 - MH_entropy_sigma_R1, MH_entropy_exp_R1 +
    #                  MH_entropy_sigma_R1, color='purple', alpha=0.2)
    plt.plot(JLP_entropy_exp_R2, color='blue', label='JLP R2')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R2 - JLP_entropy_sigma_R2, JLP_entropy_exp_R2 +
                     JLP_entropy_sigma_R2, color='blue', alpha=0.2)
    # plt.plot(MH_entropy_exp_R2, color='red', label='MH R2')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R2 - MH_entropy_sigma_R2, MH_entropy_exp_R2 +
    #                  MH_entropy_sigma_R2, color='red', alpha=0.2)
    # plt.plot(JLP_entropy_exp_R3, color='cyan', label='JLP R3')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R3 - JLP_entropy_sigma_R3, JLP_entropy_exp_R3 +
    #                  JLP_entropy_sigma_R3, color='cyan', alpha=0.2)
    # plt.plot(MH_entropy_exp_R3, color='orange', label='MH R3')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R3 - MH_entropy_sigma_R3, MH_entropy_exp_R3 +
    #                  MH_entropy_sigma_R2, color='orange', alpha=0.2)
    #plt.title(r'Average entropy of $b[\lambda]$ over time')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.grid(True)
    plt.xlim(0, 20)
    fig.savefig('../figures/Plan_Stat_JLP_R1.eps')
    plt.show()

    # Statistical R2 comparison JLP
    fig = plt.figure(1008)
    plt.plot(JLP_e_entropy_exp_R1, color='black', label='JLP R1')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_e_entropy_exp_R1 - JLP_e_entropy_sigma_R1, JLP_e_entropy_exp_R1 +
                     JLP_e_entropy_sigma_R1, color='black', alpha=0.2)
    # plt.plot(MH_entropy_exp_R1, color='purple', label='MH R1')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R1 - MH_entropy_sigma_R1, MH_entropy_exp_R1 +
    #                  MH_entropy_sigma_R1, color='purple', alpha=0.2)
    plt.plot(JLP_e_entropy_exp_R2, color='blue', label='JLP R2')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_e_entropy_exp_R2 - JLP_e_entropy_sigma_R2, JLP_e_entropy_exp_R2 +
                     JLP_e_entropy_sigma_R2, color='blue', alpha=0.2)
    # plt.plot(MH_entropy_exp_R2, color='red', label='MH R2')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R2 - MH_entropy_sigma_R2, MH_entropy_exp_R2 +
    #                  MH_entropy_sigma_R2, color='red', alpha=0.2)
    # plt.plot(JLP_entropy_exp_R3, color='cyan', label='JLP R3')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R3 - JLP_entropy_sigma_R3, JLP_entropy_exp_R3 +
    #                  JLP_entropy_sigma_R3, color='cyan', alpha=0.2)
    # plt.plot(MH_entropy_exp_R3, color='orange', label='MH R3')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R3 - MH_entropy_sigma_R3, MH_entropy_exp_R3 +
    #                  MH_entropy_sigma_R2, color='orange', alpha=0.2)
    #plt.title(r'Average entropy of $b[\lambda]$ over time')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.grid(True)
    plt.xlim(0, 20)
    fig.savefig('../figures/Plan_Stat_JLP_R2.eps')
    plt.show()

    # Statistical R1 comparison MH
    if activate_MH is True:
        fig = plt.figure(1002)
        # plt.plot(JLP_entropy_exp_R1, color='black', label='JLP R1')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R1 - JLP_entropy_sigma_R1, JLP_entropy_exp_R1 +
        #                  JLP_entropy_sigma_R1, color='black', alpha=0.2)
        plt.plot(MH_entropy_exp_R1, color='purple', label='MH R1')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R1 - MH_entropy_sigma_R1, MH_entropy_exp_R1 +
                         MH_entropy_sigma_R1, color='purple', alpha=0.2)
        # plt.plot(JLP_entropy_exp_R2, color='blue', label='JLP R2')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R2 - JLP_entropy_sigma_R2, JLP_entropy_exp_R2 +
        #                  JLP_entropy_sigma_R2, color='blue', alpha=0.2)
        plt.plot(MH_entropy_exp_R2, color='red', label='MH R2')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R2 - MH_entropy_sigma_R2, MH_entropy_exp_R2 +
                         MH_entropy_sigma_R2, color='red', alpha=0.2)
        # plt.plot(JLP_entropy_exp_R3, color='cyan', label='JLP R3')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R3 - JLP_entropy_sigma_R3, JLP_entropy_exp_R3 +
        #                  JLP_entropy_sigma_R3, color='cyan', alpha=0.2)
        # plt.plot(MH_entropy_exp_R3, color='orange', label='MH R3')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R3 - MH_entropy_sigma_R3, MH_entropy_exp_R3 +
        #                  MH_entropy_sigma_R2, color='orange', alpha=0.2)
        #plt.title(r'Average entropy of $b[\lambda]$ over time')
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_Stat_MH_R1.eps')
        plt.show()

    # Statistical R2 comparison MH
        fig = plt.figure(1009)
        # plt.plot(JLP_entropy_exp_R1, color='black', label='JLP R1')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R1 - JLP_entropy_sigma_R1, JLP_entropy_exp_R1 +
        #                  JLP_entropy_sigma_R1, color='black', alpha=0.2)
        plt.plot(MH_e_entropy_exp_R1, color='purple', label='MH R1')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_e_entropy_exp_R1 - MH_e_entropy_sigma_R1, MH_e_entropy_exp_R1 +
                         MH_e_entropy_sigma_R1, color='purple', alpha=0.2)
        # plt.plot(JLP_entropy_exp_R2, color='blue', label='JLP R2')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R2 - JLP_entropy_sigma_R2, JLP_entropy_exp_R2 +
        #                  JLP_entropy_sigma_R2, color='blue', alpha=0.2)
        plt.plot(MH_e_entropy_exp_R2, color='red', label='MH R2')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_e_entropy_exp_R2 - MH_e_entropy_sigma_R2, MH_e_entropy_exp_R2 +
                         MH_e_entropy_sigma_R2, color='red', alpha=0.2)
        # plt.plot(JLP_entropy_exp_R3, color='cyan', label='JLP R3')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R3 - JLP_entropy_sigma_R3, JLP_entropy_exp_R3 +
        #                  JLP_entropy_sigma_R3, color='cyan', alpha=0.2)
        # plt.plot(MH_entropy_exp_R3, color='orange', label='MH R3')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R3 - MH_entropy_sigma_R3, MH_entropy_exp_R3 +
        #                  MH_entropy_sigma_R2, color='orange', alpha=0.2)
        #plt.title(r'Average entropy of $b[\lambda]$ over time')
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_Stat_MH_R2.eps')
        plt.show()

    # MSDE comparison all
    if activate_MH is True:
        fig = plt.figure(200)
        for idx in range(number_of_seeds):
            plt.plot(JLP_MSDE_R1[idx], color='black', alpha=1 / number_of_seeds, label='JLP R1')
            plt.plot(MH_MSDE_R1[idx], color='purple', alpha=1 / number_of_seeds, label='MH R1')
            plt.plot(JLP_MSDE_R2[idx], color='blue', alpha=1 / number_of_seeds, label='JLP R2')
            plt.plot(MH_MSDE_R2[idx], color='red', alpha=1 / number_of_seeds, label='MH R2')
            plt.plot(WEU_MSDE[idx], color='green', alpha=1 / number_of_seeds, label='WEU')
        #plt.title('Average MSDE over time')
        plt.xlabel('Time step')
        plt.ylabel('MSDE')
        plt.legend(loc='upper left')
        #plt.yscale('log')
        plt.tight_layout()
        plt.grid(True)
        #plt.ylim(0, 0.4)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_Fixed_MSDE.eps')
        plt.show()

    JLP_MSDE_sigma_R1 = compute_sigma_vec(JLP_MSDE_R1, JLP_MSDE_exp_R1)
    JLP_MSDE_sigma_R2 = compute_sigma_vec(JLP_MSDE_R2, JLP_MSDE_exp_R2)
    WEU_MSDE_sigma = compute_sigma_vec(WEU_MSDE, WEU_MSDE_exp)

    if activate_MH is True:
        MH_MSDE_sigma_R1 = compute_sigma_vec(MH_MSDE_R1, MH_MSDE_exp_R1)
        MH_MSDE_sigma_R2 = compute_sigma_vec(MH_MSDE_R2, MH_MSDE_exp_R2)

    # MSDE comparison R1
    if activate_MH is True:
        fig = plt.figure(201)
        plt.plot(JLP_MSDE_exp_R1, color='black', label='JLP R1')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R1 - JLP_MSDE_sigma_R1, JLP_MSDE_exp_R1 +
                          JLP_MSDE_sigma_R1, color='black', alpha=0.2)
        plt.plot(MH_MSDE_exp_R1, color='purple', label='MH R1')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R1 - MH_MSDE_sigma_R1, MH_MSDE_exp_R1 +
                         MH_MSDE_sigma_R1, color='purple', alpha=0.2)
        # plt.plot(JLP_MSDE_exp_R2, color='blue', label='JLP')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R2 - JLP_MSDE_sigma_R2, JLP_MSDE_exp_R2 +
        #                  JLP_MSDE_sigma_R2, color='blue', alpha=0.2)
        # plt.plot(MH_MSDE_exp_R2, color='red', label='MH')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R2 - MH_MSDE_sigma_R2, MH_MSDE_exp_R2 +
        #                  MH_MSDE_sigma_R2, color='red', alpha=0.2)
        # plt.plot(JLP_MSDE_exp_R3, color='cyan', label='JLP')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R3 - JLP_MSDE_sigma_R3, JLP_MSDE_exp_R3 +
        #                  JLP_MSDE_sigma_R3, color='cyan', alpha=0.2)
        # plt.plot(MH_MSDE_exp_R3, color='orange', label='MH')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R3 - MH_MSDE_sigma_R3, MH_MSDE_exp_R3 +
        #                 MH_MSDE_sigma_R3, color='orange', alpha=0.2)
        plt.plot(WEU_MSDE_exp, color='green', label='WEU')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), WEU_MSDE_exp - WEU_MSDE_sigma, WEU_MSDE_exp +
                         WEU_MSDE_sigma, color='green', alpha=0.2)
        #plt.title('Average MSDE over time')
        plt.xlabel('Time step')
        plt.ylabel('MSDE')
        plt.legend(loc='upper right')
        #plt.yscale('log')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_' + fixed_stat + '_MSDE_R1.eps')
        plt.show()

        # Statistical MSDE comparison R2
        fig = plt.figure(208)
        # plt.plot(JLP_MSDE_exp_R1, color='black', label='JLP R1')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R1 - JLP_MSDE_sigma_R1, JLP_MSDE_exp_R1 +
        #                   JLP_MSDE_sigma_R1, color='black', alpha=0.2)
        # plt.plot(MH_MSDE_exp_R1, color='purple', label='MH R1')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R1 - MH_MSDE_sigma_R1, MH_MSDE_exp_R1 +
        #                  MH_MSDE_sigma_R1, color='purple', alpha=0.2)
        plt.plot(JLP_MSDE_exp_R2, color='blue', label='JLP R2')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R2 - JLP_MSDE_sigma_R2, JLP_MSDE_exp_R2 +
                         JLP_MSDE_sigma_R2, color='blue', alpha=0.2)
        plt.plot(MH_MSDE_exp_R2, color='red', label='MH R2')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R2 - MH_MSDE_sigma_R2, MH_MSDE_exp_R2 +
                         MH_MSDE_sigma_R2, color='red', alpha=0.2)
        # plt.plot(JLP_MSDE_exp_R3, color='cyan', label='JLP')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R3 - JLP_MSDE_sigma_R3, JLP_MSDE_exp_R3 +
        #                  JLP_MSDE_sigma_R3, color='cyan', alpha=0.2)
        # plt.plot(MH_MSDE_exp_R3, color='orange', label='MH')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R3 - MH_MSDE_sigma_R3, MH_MSDE_exp_R3 +
        #                 MH_MSDE_sigma_R3, color='orange', alpha=0.2)
        plt.plot(WEU_MSDE_exp, color='green', label='WEU')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), WEU_MSDE_exp - WEU_MSDE_sigma, WEU_MSDE_exp +
                         WEU_MSDE_sigma, color='green', alpha=0.2)
        #plt.title('Average MSDE over time')
        plt.xlabel('Time step')
        plt.ylabel('MSDE')
        plt.legend(loc='upper right')
        #plt.yscale('log')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_' + fixed_stat + '_MSDE_R2.eps')
        plt.show()

    # Statistical MSDE comparison JLP
    fig = plt.figure(203)
    plt.plot(JLP_MSDE_exp_R1, color='black', label='JLP R1')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R1 - JLP_MSDE_sigma_R1, JLP_MSDE_exp_R1 +
                     JLP_MSDE_sigma_R1, color='black', alpha=0.2)
    # plt.plot(MH_MSDE_exp_R1, color='purple', label='MH')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R1 - MH_MSDE_sigma_R1, MH_MSDE_exp_R1 +
    #                  MH_MSDE_sigma_R1, color='purple', alpha=0.2)
    plt.plot(JLP_MSDE_exp_R2, color='blue', label='JLP R2')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R2 - JLP_MSDE_sigma_R2, JLP_MSDE_exp_R2 +
                     JLP_MSDE_sigma_R2, color='blue', alpha=0.2)
    # plt.plot(MH_MSDE_exp_R2, color='red', label='MH')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R2 - MH_MSDE_sigma_R2, MH_MSDE_exp_R2 +
    #                 MH_MSDE_sigma_R2, color='red', alpha=0.2)
    # plt.plot(JLP_MSDE_exp_R3, color='cyan', label='JLP R3')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R3 - JLP_MSDE_sigma_R3, JLP_MSDE_exp_R3 +
    #                   JLP_MSDE_sigma_R3, color='cyan', alpha=0.2)
    # plt.plot(MH_MSDE_exp_R3, color='orange', label='MH')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R3 - MH_MSDE_sigma_R3, MH_MSDE_exp_R3 +
    #                 MH_MSDE_sigma_R3, color='orange', alpha=0.2)
    plt.plot(WEU_MSDE_exp, color='green', label='WEU')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), WEU_MSDE_exp - WEU_MSDE_sigma, WEU_MSDE_exp +
                      WEU_MSDE_sigma, color='green', alpha=0.2)
    #plt.title('Average MSDE over time')
    plt.xlabel('Time step')
    plt.ylabel('MSDE')
    plt.legend(loc='upper right')
    #plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)
    #plt.ylim(0, 0.4)
    plt.xlim(0, 20)
    fig.savefig('../figures/Plan_Stat_MSDE_JLP.eps')
    plt.show()

    # Statistical MSDE comparison MH
    if activate_MH is True:
        # R1 R2 R3 comparison JLP
        fig = plt.figure(204)
        # plt.plot(JLP_MSDE_exp_R1, color='black', label='JLP R1')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R1 - JLP_MSDE_sigma_R1, JLP_MSDE_exp_R1 +
        #                  JLP_MSDE_sigma_R1, color='black', alpha=0.2)
        plt.plot(MH_MSDE_exp_R1, color='purple', label='MH R1')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R1 - MH_MSDE_sigma_R1, MH_MSDE_exp_R1 +
                         MH_MSDE_sigma_R1, color='purple', alpha=0.2)
        # plt.plot(JLP_MSDE_exp_R2, color='blue', label='JLP R2')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R2 - JLP_MSDE_sigma_R2, JLP_MSDE_exp_R2 +
        #                  JLP_MSDE_sigma_R2, color='blue', alpha=0.2)
        plt.plot(MH_MSDE_exp_R2, color='red', label='MH R2')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R2 - MH_MSDE_sigma_R2, MH_MSDE_exp_R2 +
                        MH_MSDE_sigma_R2, color='red', alpha=0.2)
        # plt.plot(JLP_MSDE_exp_R3, color='cyan', label='JLP R3')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R3 - JLP_MSDE_sigma_R3, JLP_MSDE_exp_R3 +
        #                   JLP_MSDE_sigma_R3, color='cyan', alpha=0.2)
        # plt.plot(MH_MSDE_exp_R3, color='orange', label='MH')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R3 - MH_MSDE_sigma_R3, MH_MSDE_exp_R3 +
        #                 MH_MSDE_sigma_R3, color='orange', alpha=0.2)
        plt.plot(WEU_MSDE_exp, color='green', label='WEU')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), WEU_MSDE_exp - WEU_MSDE_sigma, WEU_MSDE_exp +
                          WEU_MSDE_sigma, color='green', alpha=0.2)
        #plt.title('Average MSDE over time')
        plt.xlabel('Time step')
        plt.ylabel('MSDE')
        plt.legend(loc='upper right')
        #plt.yscale('log')
        plt.tight_layout()
        plt.grid(True)
        #plt.ylim(0, 0.4)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_Stat_MSDE_MH.eps')
        plt.show()

    if activate_MH is True:
        fig = plt.figure(400)
        idx_time = 0
        key_color = {'JLP_R1': 'black', 'MH_R1': 'purple', 'JLP_R2': 'blue','MH_R2': 'red', 'WEU': 'green'}
        for key in time_dict:
            idx_time += 1
            plt.plot(np.repeat(idx_time, number_of_seeds), time_dict[key], '*', color=key_color[key])
        plt.title('Scenario time')
        plt.xticks(np.arange(7) + 1, time_dict.keys())
        plt.ylabel('Time [sec]')
        plt.grid(True)
        plt.show()

    fig = plt.figure(500)
    if activate_MH is True:
        plt.plot(np.array(range(len(JLP_entropy_exp_R1))), time_dict_track['MH_R2'], color='red', label='JLP')
        plt.text(len(time_dict_track['MH_R2']) - 2, time_dict_track['MH_R2'][-1], 'MH')
    plt.plot(np.array(range(len(JLP_entropy_exp_R1))), time_dict_track['JLP_R2'], color='black', label='MH')
    plt.text(len(time_dict_track['JLP_R2']) - 2, time_dict_track['JLP_R2'][-1] -1, 'JLP')
    plt.plot(np.array(range(len(JLP_entropy_exp_R1))), time_dict_track['WEU'], color='green', label='JLP')
    plt.text(len(time_dict_track['WEU']) - 2, time_dict_track['WEU'][-1], 'WEU')
    #plt.title('Computation time of different methods with different number of beliefs')
    plt.xlabel('Time step')
    plt.ylabel('Time [s]')
    #plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(1, 20)
    fig.savefig('../figures/Plan_Time.eps')
    plt.show()


if __name__ == "__main__":
    main()
