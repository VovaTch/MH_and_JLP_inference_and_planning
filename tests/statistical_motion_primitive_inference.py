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

def motion_primitive_planning(seed, Lambda_BSP_mode, reward_mode, cls_model, number_of_beliefs, random_objects=True):

    # Initializations ----------------------------------------------------
    sample_use_flag = True
    ML_planning_flag = False
    track_length = 20
    horizon = 5
    ML_update = True
    MCTS_flag = True
    MCTS_branches = 4
    cls_enable = True
    display_graphs = True
    measurement_radius_limit = 10
    measurement_radius_minimum = 0
    com_radius = 100000
    class_GT = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1}#, 6: 2}#, 7: 1, 8: 1, 9: 2}#, 10: 1}
    np.random.seed(seed)
    measurements_enabler = True
    opening_angle = 60
    opening_angle_rad = opening_angle * math.pi / 180
    num_samp = 50
    CVaR_flag = True
    start_time = time.time()
    entropy_lower_limit = None

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

        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([2.5, -4.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(3.141/2, 0.0, 0.0), np.array([4, -7.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141/2, 0.0, 0.0), np.array([4.5, -1.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141 / 4, 0.0, 0.0), np.array([1.0, -4.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0*3.141, 0.0, 0.0), np.array([7.5, 4.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141 / 4, 0.0, 0.0), np.array([2, -2, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141 / 2, 0.0, 0.0), np.array([8.5, 4.7, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141 / 4, 0.0, 0.0), np.array([11.5, 1.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-2 * 3.141 / 4, 0.0, 0.0), np.array([12, -6.5, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-2*3.141/4, 0.0, 0.0), np.array([14.5, 2.7, 0.0])))

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
                             cls_enable=cls_enable, pruning_threshold=250, ML_update=ML_update))

        lambda_belief = LambdaBelief(belief_list)
        lambda_planner = LambdaPlannerPrimitives(lambda_belief)

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
    sigma_list = list()
    sigma_list.append(0)
    time_list = list()
    time_list.append(0)
    MSDE = list()
    MSDE.append(1/4)

    #for sequence_idx in range(track_length):
    for idx in range(track_length):

        step_time = time.time()

        # action, reward = lambda_planner.planning_session(action_noise, np.diag(geo_model_noise_diag), horizon=horizon, reward_print=True,
        #                                                  enable_MCTS=MCTS_flag, MCTS_braches_per_action=MCTS_branches,
        #                                                  ML_planning=ML_planning_flag, sample_use_flag=sample_use_flag,
        #                                                  reward_mode=reward_mode, motion_primitives=motion_primitives)
        # action = motion_primitives[0]

        if idx < 5:
            action = gtsam.Pose3(gtsam.Pose2(1.0, 0.0, -np.pi / 16))

        elif idx < 15:
            action = gtsam.Pose3(gtsam.Pose2(1.0, 0.0, np.pi / 16))

        else:
            action = gtsam.Pose3(gtsam.Pose2(1.0, 0.0, -np.pi / 16))

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
        if Lambda_BSP_mode == 1:

            for belief in lambda_belief.belief_list:
                object_list = belief.obj_realization
                break

            for obj in object_list:
                entropy_collector -= lambda_belief.entropy(obj)
                sigma_collector += 2 * np.sqrt(lambda_belief.lambda_covariance(obj)[0, 0]) / len(object_list)

        if Lambda_BSP_mode == 2:

            for obj in lambda_belief.object_lambda_dict:
                entropy_collector -= lambda_belief.\
                    lambda_entropy_individual_numeric(lambda_belief.object_lambda_dict[obj])
                sigma_collector += 2 * np.sqrt(lambda_belief.
                                               lambda_covariance_numeric(
                    lambda_belief.object_lambda_dict[obj])[0, 0]) / len(lambda_belief.object_lambda_dict)

        current_entropy_list.append(entropy_collector)
        MSDE.append(lambda_belief.MSDE_expectation(class_GT))
        sigma_list.append(sigma_collector)

        time_list.append(time.time() - step_time)

    print(chosen_action_list)

    return current_entropy_list, MSDE, sigma_list, time.time() - start_time, time_list

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

# Main function; runs the statistical study.
def main():

    number_of_seeds = 10
    number_of_beliefs = 5
    seed_offset = 270
    cls_model_1 = clsmodel_lg1.ClsModel()
    cls_model_2 = clsUmodel_fake_1d.JLPModel()
    random_object_flag = True

    if number_of_seeds == 1:
        fixed_stat = 'Fixed'
    else:
        fixed_stat = 'Stat'

    JLP_entropy_R1 = list()

    JLP_MSDE_R1 = list()

    JLP_sigma_R1 = list()

    JLP_entropy_R2 = list()

    JLP_MSDE_R2 = list()

    JLP_sigma_R2 = list()

    MH_entropy_R1 = list()

    MH_MSDE_R1 = list()

    MH_sigma_R1 = list()

    MH_entropy_R2 = list()

    MH_MSDE_R2 = list()

    MH_sigma_R2 = list()

    WEU_entropy = list()

    WEU_MSDE = list()

    time_dict = {'JLP_R1': list(), 'MH_R1': list(), 'JLP_R2': list(), 'MH_R2': list(), 'WEU': list()}

    for seed_idx in range(number_of_seeds):

        # JLP planning lambda entropy
        Lambda_BSP_mode = 2
        reward_mode = 1

        entropy, MSDE, sigma, time, _ = motion_primitive_planning(seed_idx + seed_offset, Lambda_BSP_mode, reward_mode,
                                                               cls_model_2,
                                                               number_of_beliefs=number_of_beliefs,
                                                                  random_objects=random_object_flag)
        JLP_entropy_R1.append(entropy)
        JLP_MSDE_R1.append(MSDE)
        JLP_sigma_R1.append(sigma)

        time_dict['JLP_R1'].append(time)

        if seed_idx == 0:
            JLP_entropy_exp_R1 = np.array(entropy) / number_of_seeds
            JLP_MSDE_exp_R1 = np.array(MSDE) / number_of_seeds
            JLP_sigma_exp_R1 = np.array(sigma) / number_of_seeds
        else:
            JLP_entropy_exp_R1 += np.array(entropy) / number_of_seeds
            JLP_MSDE_exp_R1 += np.array(MSDE) / number_of_seeds
            JLP_sigma_exp_R1 += np.array(sigma) / number_of_seeds

        # JLP planning lambda expectation entropy
        Lambda_BSP_mode = 2
        reward_mode = 2

        entropy, MSDE, sigma, time, _ = motion_primitive_planning(seed_idx + seed_offset, Lambda_BSP_mode, reward_mode,
                                                               cls_model_2,
                                                               number_of_beliefs=number_of_beliefs,
                                                                  random_objects=random_object_flag)
        JLP_entropy_R2.append(entropy)
        JLP_MSDE_R2.append(MSDE)
        JLP_sigma_R2.append(sigma)

        time_dict['JLP_R2'].append(time)

        if seed_idx == 0:
            JLP_entropy_exp_R2 = np.array(entropy) / number_of_seeds
            JLP_MSDE_exp_R2 = np.array(MSDE) / number_of_seeds
            JLP_sigma_exp_R2 = np.array(sigma) / number_of_seeds
        else:
            JLP_entropy_exp_R2 += np.array(entropy) / number_of_seeds
            JLP_MSDE_exp_R2 += np.array(MSDE) / number_of_seeds
            JLP_sigma_exp_R2 += np.array(sigma) / number_of_seeds

        # MH planning lambda entropy
        Lambda_BSP_mode = 1
        reward_mode = 1

        entropy, MSDE, sigma, time, _ = motion_primitive_planning(seed_idx + seed_offset, Lambda_BSP_mode, reward_mode,
                                                               cls_model_1,
                                                               number_of_beliefs=number_of_beliefs,
                                                                  random_objects=random_object_flag)
        MH_entropy_R1.append(entropy)
        MH_MSDE_R1.append(MSDE)
        MH_sigma_R1.append(sigma)

        time_dict['MH_R1'].append(time)

        if seed_idx == 0:
            MH_entropy_exp_R1 = np.array(entropy) / number_of_seeds
            MH_MSDE_exp_R1 = np.array(MSDE) / number_of_seeds
            MH_sigma_exp_R1 = np.array(sigma) / number_of_seeds
        else:
            MH_entropy_exp_R1 += np.array(entropy) / number_of_seeds
            MH_MSDE_exp_R1 += np.array(MSDE) / number_of_seeds
            MH_sigma_exp_R1 += np.array(sigma) / number_of_seeds

        # MH planning lambda expectation entropy
        Lambda_BSP_mode = 1
        reward_mode = 2

        entropy, MSDE, sigma, time, _ = motion_primitive_planning(seed_idx + seed_offset, Lambda_BSP_mode, reward_mode,
                                                               cls_model_1,
                                                               number_of_beliefs=number_of_beliefs,
                                                                  random_objects=random_object_flag)
        MH_entropy_R2.append(entropy)
        MH_MSDE_R2.append(MSDE)
        MH_sigma_R2.append(sigma)

        time_dict['MH_R2'].append(time)

        if seed_idx == 0:
            MH_entropy_exp_R2 = np.array(entropy) / number_of_seeds
            MH_MSDE_exp_R2 = np.array(MSDE) / number_of_seeds
            MH_sigma_exp_R2 = np.array(sigma) / number_of_seeds
        else:
            MH_entropy_exp_R2 += np.array(entropy) / number_of_seeds
            MH_MSDE_exp_R2 += np.array(MSDE) / number_of_seeds
            MH_sigma_exp_R2 += np.array(sigma) / number_of_seeds

        # Patten18arj like planning
        entropy, MSDE, _, time, _ = motion_primitive_planning(seed_idx + seed_offset, Lambda_BSP_mode, reward_mode,
                                                           cls_model_1,
                                                           number_of_beliefs=1,
                                                                  random_objects=random_object_flag)
        WEU_entropy.append(entropy)
        WEU_MSDE.append(MSDE)

        time_dict['WEU'].append(time)

        if seed_idx == 0:
            WEU_MSDE_exp = np.array(MSDE) / number_of_seeds
        else:
            WEU_MSDE_exp += np.array(MSDE) / number_of_seeds

    time_dict_track = {'5bel_MH':None, '10bel_MH':None, '15bel_MH':None, '20bel_MH':None, '25bel_MH':None, 'JLP':None,
                 'WEU':None}
    for time_idx in range(5):

        number_of_beliefs_time = (time_idx + 1) * 5
        name = str(number_of_beliefs_time) + 'bel_MH'
        _, _, _, _, time_dict_track[name] = motion_primitive_planning(seed_idx, 1, 1,
                                                               cls_model_1,
                                                               number_of_beliefs=number_of_beliefs_time,
                                                                  random_objects=random_object_flag)
    _, _, _, _, time_dict_track['JLP'] = motion_primitive_planning(seed_idx, 2, 1,
                                                            cls_model_2,
                                                            number_of_beliefs=number_of_beliefs_time,
                                                                  random_objects=random_object_flag)
    _, _, _, _, time_dict_track['WEU'] = motion_primitive_planning(seed_idx, 1, 1,
                                                            cls_model_1,
                                                            number_of_beliefs=1,
                                                                  random_objects=random_object_flag)

    fig = plt.figure(000)
    for idx in range(5):
        name = str((idx + 1) * 5) + 'bel_MH'
        plt.plot(np.array(range(len(JLP_entropy_exp_R1))), time_dict_track[name], color='red', label='JLP')
        plt.text(len(time_dict_track[name]) - 2, time_dict_track[name][-1], r'$N_b$=' + str((idx + 1) * 5))
    plt.plot(np.array(range(len(JLP_entropy_exp_R1))), time_dict_track['JLP'], color='black', label='MH')
    plt.text(len(time_dict_track['JLP']) - 2, time_dict_track['JLP'][-1] -1, 'JLP')
    plt.plot(np.array(range(len(JLP_entropy_exp_R1))), time_dict_track['WEU'], color='green', label='JLP')
    plt.text(len(time_dict_track['WEU']) - 2, time_dict_track['WEU'][-1], r'$N_b$=1')
    #plt.title('Computation time of different methods with different number of beliefs')
    plt.xlabel('Time step')
    plt.ylabel('Time [s]')
    #plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(1, 20)
    plt.savefig('../figures/Inf_' + fixed_stat + '_Time.eps')
    plt.show()

    fig = plt.figure(100)
    for idx in range(number_of_seeds):
        plt.plot(JLP_entropy_R1[idx], color='black', alpha=1 / number_of_seeds, label='JLP')
        plt.plot(MH_entropy_R1[idx], color='purple', alpha=1 / number_of_seeds, label='MH')
        plt.plot(JLP_entropy_R2[idx], color='blue', alpha=1 / number_of_seeds, label='JLP')
        plt.plot(MH_entropy_R2[idx], color='red', alpha=1 / number_of_seeds, label='MH')
    plt.title(r'Entropy reward of $b[\lambda]$ over time')
    plt.xlabel('Time step')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('../figures/Inf_Fixed_R1.eps')
    plt.show()

    JLP_entropy_sigma_R1 = compute_sigma_vec(JLP_entropy_R1, JLP_entropy_exp_R1)
    MH_entropy_sigma_R1 = compute_sigma_vec(MH_entropy_R1, MH_entropy_exp_R1)
    JLP_entropy_sigma_R2 = compute_sigma_vec(JLP_entropy_R2, JLP_entropy_exp_R2)
    MH_entropy_sigma_R2 = compute_sigma_vec(MH_entropy_R2, MH_entropy_exp_R2)

    fig = plt.figure(101)
    plt.plot(JLP_entropy_exp_R1, color='black', label='JLP')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R1 - JLP_entropy_sigma_R1, JLP_entropy_exp_R1 +
                     JLP_entropy_sigma_R1, color='black', alpha=0.2)
    plt.plot(MH_entropy_exp_R1, color='purple', label='MH')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R1 - MH_entropy_sigma_R1, MH_entropy_exp_R1 +
                     MH_entropy_sigma_R1, color='purple', alpha=0.2)
    plt.plot(JLP_entropy_exp_R2, color='blue', label='JLP')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R2 - JLP_entropy_sigma_R2, JLP_entropy_exp_R2 +
                     JLP_entropy_sigma_R2, color='blue', alpha=0.2)
    plt.plot(MH_entropy_exp_R2, color='red', label='MH')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R2 - MH_entropy_sigma_R2, MH_entropy_exp_R2 +
                     MH_entropy_sigma_R2, color='red', alpha=0.2)
    #plt.title(r'Average entropy of $b[\lambda]$ over time')
    plt.xlabel('Time step')
    plt.ylabel('Reward')
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.savefig('../figures/Inf_Stat_R1.eps')
    plt.show()

    fig = plt.figure(200)
    for idx in range(number_of_seeds):
        #plt.plot(JLP_MSDE_R1[idx], color='black', alpha=1 / number_of_seeds, label='JLP')
        #plt.plot(MH_MSDE_R1[idx], color='purple', alpha=1 / number_of_seeds, label='MH')
        plt.plot(JLP_MSDE_R2[idx], color='blue', alpha=1 / number_of_seeds, label='JLP')
        plt.plot(MH_MSDE_R2[idx], color='red', alpha=1 / number_of_seeds, label='MH')
        plt.plot(WEU_MSDE[idx], color='green', alpha=1 / number_of_seeds, label='WEU')
    plt.title('Average MSDE over time')
    plt.xlabel('Time step')
    plt.ylabel('MSDE')
    plt.legend(loc='lower left')
    plt.ylim(0, 1)
    plt.xlim(0, 20)
    plt.grid(True)
    plt.savefig('../figures/Inf_Fixed_MSDE.eps')
    plt.show()

    JLP_MSDE_sigma_R1 = compute_sigma_vec(JLP_MSDE_R1, JLP_MSDE_exp_R1)
    MH_MSDE_sigma_R1 = compute_sigma_vec(MH_MSDE_R1, MH_MSDE_exp_R1)
    JLP_MSDE_sigma_R2 = compute_sigma_vec(JLP_MSDE_R2, JLP_MSDE_exp_R2)
    MH_MSDE_sigma_R2 = compute_sigma_vec(MH_MSDE_R2, MH_MSDE_exp_R2)
    WEU_MSDE_sigma = compute_sigma_vec(WEU_MSDE, WEU_MSDE_exp)

    fig = plt.figure(201)
    # plt.plot(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R1, color='black', label='JLP')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R1 - JLP_MSDE_sigma_R1, JLP_MSDE_exp_R1 +
    #                  JLP_MSDE_sigma_R1, color='black', alpha=0.2)
    # plt.plot(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R1, color='purple', label='MH')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R1 - MH_MSDE_sigma_R1, MH_MSDE_exp_R1 +
    #                  MH_MSDE_sigma_R1, color='purple', alpha=0.2)
    plt.plot(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R2, color='blue', label='JLP')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R2 - JLP_MSDE_sigma_R2, JLP_MSDE_exp_R2 +
                     JLP_MSDE_sigma_R2, color='blue', alpha=0.2)
    plt.plot(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R2, color='red', label='MH')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_MSDE_exp_R2 - MH_MSDE_sigma_R2, MH_MSDE_exp_R2 +
                     MH_MSDE_sigma_R2, color='red', alpha=0.2)
    plt.plot(np.array(range(len(JLP_entropy_exp_R1))), WEU_MSDE_exp, color='green', label='WEU')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), WEU_MSDE_exp - WEU_MSDE_sigma, WEU_MSDE_exp +
                     WEU_MSDE_sigma, color='green', alpha=0.2)
    #plt.title('Average MSDE over time')
    plt.xlabel('Time step')
    plt.ylabel('MSDE')
    #plt.yscale('log')
    plt.legend(loc='lower left')
    plt.ylim(0, 0.4)
    plt.xlim(0, 20)
    plt.grid(True)
    plt.savefig('../figures/Inf_Stat_MSDE.eps')
    plt.show()

    fig = plt.figure(400)
    idx_time = 0
    key_color = {'JLP_R1': 'black', 'MH_R1': 'purple', 'JLP_R2': 'blue','MH_R2': 'red', 'WEU': 'green'}
    for key in time_dict:
        idx_time += 1
        plt.plot(np.repeat(idx_time, number_of_seeds), time_dict[key], '*', color=key_color[key])
    #plt.title('Scenario time')
    plt.xticks(np.arange(5) + 1, time_dict.keys())
    plt.ylabel('Time [sec]')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()
