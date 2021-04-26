from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.io
from scipy import special
from scipy import stats
import gtsam
import math
import damodel
import geomodel_dual as geomodel_proj
import plotterdaac2d
import gaussianb_jlp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from JLP_planner_AVD import JLPPLannerPrimitives
from hybridb import HybridBelief
from lambdab_lg import LambdaBelief
from lambda_planner_AVD import LambdaPlannerPrimitives
#import clsUmodel_real_4d
# import time
import json
import time
import clsmodel_real_lg4 as clsmodel_lg1
import clsUmodel_real_4d as clsUmodel_fake_1d
from PIL import Image

def motion_primitive_planning(seed, Lambda_BSP_mode, reward_mode, cls_model, number_of_beliefs,
                              random_objects=True, use_bounds = False, plt_ax = None, bar_ax = None,
                              initial_image = '000510000350101.jpg'):


    ## --------------- PRLIMINARIES: NETWORK AND SCENARIO SETUP

    # Find the corresponding depth image to the jpg
    def depth_image_find(image_address):

        depth_address = image_address[0:-5]
        depth_address += '3.png'
        return depth_address

    # Load json file
    actions = ['forward','rotate_ccw','rotate_cw','backward','left','right']

    scenario_file = open('home_005_data/annotations.json')
    scenario_data = json.load(scenario_file)

    classification_file = open('home_005_data/classification_data_vgg.json')
    classification_data = json.load(classification_file)

    #------------------------------------- EXTRACT GROUND TRUTH

    detailed_GT_file = scipy.io.loadmat('home_005_data/image_structs.mat')

    GT_pose = dict()
    x_pos = list()
    y_pos = list()
    z_pos = list()

    for location in detailed_GT_file['image_structs']['world_pos'][0]:
        x_pos.append(float(location[0]))
        y_pos.append(float(location[2]))
        z_pos.append(float(location[1]))

    psi = list()
    theta = list()
    for rotation in detailed_GT_file['image_structs']['direction'][0]:
        psi.append(math.atan2(rotation[2], rotation[0]) * 180 / np.pi)
        theta.append(math.atan2(rotation[1], np.sqrt(rotation[2]**2 + rotation[0]**2)) * 180 / np.pi)

    idx = int(0)
    for name in detailed_GT_file['image_structs']['image_name'][0]:
        GT_pose[name[0]] = gtsam.Pose3(gtsam.Rot3.Ypr(psi[idx] * np.pi / 180, theta[idx] * np.pi / 180, 0.0),
                                    np.array([x_pos[idx], y_pos[idx], z_pos[idx]]))
        idx += 1

    scenario_file.close()
    classification_file.close()

    # Initializations ----------------------------------------------------
    conf_flag = False
    sample_use_flag = True
    ML_planning_flag = True
    track_length = 20
    horizon = 1
    ML_update = True
    MCTS_flag = True
    MCTS_branches = 3
    cls_enable = True
    display_graphs = True
    measurement_radius_limit = 10
    measurement_radius_minimum = 0
    com_radius = 100000
    # class_GT = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2}#, 10: 1}
    # GT_cls_realization = ((1, 1), (2, 2), (3, 1), (4, 1), (5, 1), (6, 2), (7, 1), (8, 1))
    np.random.seed(seed)
    measurements_enabler = True
    opening_angle = 50
    opening_angle_rad = opening_angle * math.pi / 180
    num_samp = 50
    CVaR_flag = True
    start_time = time.time()
    entropy_lower_limit = -500
    random_object_location_flag = False

    # Set models ---------------------------------------------------------

    np.set_printoptions(precision=3)
    plt.rcParams.update({'font.size': 16})

    prior_lambda = (1, 1, 1, 1, 1)
    lambda_prior = np.array([special.digamma(prior_lambda[0]) - special.digamma(prior_lambda[4]),
                             special.digamma(prior_lambda[1]) - special.digamma(prior_lambda[4]),
                             special.digamma(prior_lambda[2]) - special.digamma(prior_lambda[4]),
                             special.digamma(prior_lambda[3]) - special.digamma(prior_lambda[4])]) #
    lambda_prior_noise_diag = np.array([special.polygamma(1, prior_lambda[0]) + special.polygamma(1, prior_lambda[4]),
                                       special.polygamma(1, prior_lambda[1]) + special.polygamma(1, prior_lambda[4]),
                                       special.polygamma(1, prior_lambda[2]) + special.polygamma(1, prior_lambda[4]),
                                       special.polygamma(1, prior_lambda[3]) + special.polygamma(1, prior_lambda[4])])
    lambda_prior_noise = gtsam.noiseModel.Diagonal.Variances(lambda_prior_noise_diag)

    da_model = damodel.DAModel(camera_fov_angle_horizontal=opening_angle_rad, range_limit=measurement_radius_limit,
                               range_minimum=measurement_radius_minimum)
    geo_model_noise_diag = np.array([0.00001, 0.00001, 30.001, 0.05, 0.05, 0.05])
    geo_model_noise = gtsam.noiseModel.Diagonal.Variances(geo_model_noise_diag)
    geo_model = geomodel_proj.GeoModel(geo_model_noise)

    # Robot poses GT
    current_image = initial_image

    GT_poses = list()
    GT_poses.append(GT_pose[initial_image])

    #------------------------------------------------ PRIORS

    prior = GT_poses[0]
    prior_noise_diag = np.array([00.000001, 0.000001, 0.0000038, 00.00002, 00.0000202, 0.000001])
    prior_noise = gtsam.noiseModel.Diagonal.Variances(prior_noise_diag)
    prior_noise_cov = np.diag(prior_noise_diag)

    # Create belief based on the switch
    if Lambda_BSP_mode == 1: #TODO: complete this side

        belief_list = list()
        lambda_planner_list = list()

        for idx in range(number_of_beliefs):
            # cls_prior_rand_0 = np.random.dirichlet(prior_lambda)
            cls_prior_rand_0_lg = np.random.multivariate_normal(lambda_prior, np.diag(lambda_prior_noise_diag))
            cls_prior_rand_0_con = np.exp(cls_prior_rand_0_lg) / (1 + np.sum(np.exp(cls_prior_rand_0_lg)))
            cls_prior_rand_0 = np.concatenate((cls_prior_rand_0_con, 1 /
                                               (1 + np.sum(np.exp(cls_prior_rand_0_lg)))), axis=None)
            cls_prior_rand = cls_prior_rand_0.tolist()


            belief_list.append(
                HybridBelief(5, geo_model, da_model, cls_model, cls_prior_rand, GT_poses[0],
                             prior_noise,
                             cls_enable=cls_enable, pruning_threshold=3.5, ML_update=ML_update))

        lambda_belief = LambdaBelief(belief_list)
        lambda_planner = LambdaPlannerPrimitives(lambda_belief, entropy_lower_limit=entropy_lower_limit,
                                                 AVD_mapping=scenario_data, AVD_poses=GT_pose)

    if Lambda_BSP_mode == 2:

        lambda_belief = gaussianb_jlp.JLPBelief(geo_model, da_model, cls_model, GT_poses[0], prior_noise,
                                                              lambda_prior_mean=lambda_prior,
                                                              lambda_prior_noise=np.diag(lambda_prior_noise_diag),
                                                              cls_enable=cls_enable)
        lambda_planner = JLPPLannerPrimitives(lambda_belief, entropy_lower_limit=entropy_lower_limit,
                                              AVD_mapping=scenario_data, AVD_poses=GT_pose)

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

    prior_entropy = lambda_entropy_individual_numeric(lambda_prior, np.diag(lambda_prior_noise_diag))

    # INFERENCE
    action_noise_diag = np.array([0.0003, 0.0003, 0.0001, 0.0003, 0.00030, 0.001])
    action_noise = gtsam.noiseModel.Diagonal.Variances(action_noise_diag)

    class_GT = {15: 2, 4: 3, 14: 2, 3: 2, 17: 3, 19: 2, 23:5, 20: 2, 29: 2, 2: 2, 5: 2}
    bars_list_of_objects = [15, 4, 14, 3, 17, 19, 23, 20, 29, 2, 10, 2, 5]

    reward_collections = list()
    chosen_action_list = list()
    current_entropy_list = list()
    current_entropy_list.append(0)
    current_R2_list = list()
    current_R2_list.append(-len(class_GT) * np.log(0.5))
    sigma_list = list()
    sigma_list.append(0)
    time_list = list()
    time_list.append(0)
    MSDE = list()
    MSDE.append(4/25)
    current_entropy_list_per_object = dict()
    current_MSDE_list_per_object = dict()
    for obj in class_GT:
        current_entropy_list_per_object[obj] = list()
        current_entropy_list_per_object[obj].append(0)
        current_MSDE_list_per_object[obj] = list()
        current_MSDE_list_per_object[obj].append(4/25)
    MSDE_per_object = list()
    sigma_list = list()


    #for sequence_idx in range(track_length):
    for idx in range(track_length):

        step_time = time.time()

        action_name, reward = lambda_planner.planning_session(action_noise, np.diag(geo_model_noise_diag), horizon=horizon,
                                                         reward_print=True,
                                                         enable_MCTS=MCTS_flag, MCTS_braches_per_action=MCTS_branches,
                                                         ML_planning=ML_planning_flag, sample_use_flag=sample_use_flag,
                                                         reward_mode=reward_mode, return_index_flag=True,
                                                         number_of_samples=number_of_beliefs,
                                                         use_lower_bound=use_bounds,
                                                         current_location_name=current_image)

        action = GT_pose[current_image].between(GT_pose[scenario_data[current_image][action_name]])
        action = action_generator(action, action_noise_diag, no_noise=True)

        # action = motion_primitives[0]
        print('----------------------------------------------------------------------------')
        print('Action:\n' + action_name)
        print('idx ' + str(idx))
        GT_poses.append(GT_poses[idx].compose(action))
        lambda_belief.action_step(action, action_noise)
        current_image = scenario_data[current_image][action_name]

        # Create the ground truth data association
        GT_DA = list()
        geo_measurements = list()
        sem_measurements = list()
        sem_measurements_exp = list()
        sem_measurements_cov = list()
        idx_range = 0
        for angle in classification_data[current_image]['Angles']:
            if classification_data[current_image]['Range'][idx_range] != 0 and classification_data[current_image]['DA'][idx_range][0] != 10: #TODO:SECOND TEMPORARY CONDITION
                GT_DA.append(classification_data[current_image]['DA'][idx_range][0])
                # geo_measurements.append([angle[0], angle[1], classification_data[current_image]['Range'][idx_range]])
                geo_measurements.append([angle[0], -0.0, classification_data[current_image]['Range'][idx_range]])

                if Lambda_BSP_mode == 1:
                    sem_measurements.append(classification_data[current_image]['CPV'][idx_range])#TODO: If MH needs more than 1 measurement change it

                if Lambda_BSP_mode == 2:
                    sem_measurements_exp.append(np.array(classification_data[current_image]['LG expectation'][idx_range]))
                    sem_measurements_cov.append(cls_model.unflatten_cov(np.array(classification_data[current_image]
                                                                                  ['LG covariance'][idx_range])))

            idx_range += 1

        sem_measurements = [sem_measurements]

        if Lambda_BSP_mode == 1:
            lambda_belief.add_measurements(geo_measurements, sem_measurements, da_current_step=GT_DA,
                                                              number_of_samples=num_samp, new_input_object_prior=None,
                                                              new_input_object_covariance=None)

        if Lambda_BSP_mode == 2:

            lambda_belief.add_measurements(geo_measurements, sem_measurements_exp, sem_measurements_cov,
                                                              GT_DA)

        lambda_belief.print_lambda_lg_params()

        for obj in class_GT:
            current_entropy_list_per_object[obj].append(0)
            current_MSDE_list_per_object[obj].append(4/25)

        entropy_collector = 0
        sigma_collector = 0
        if Lambda_BSP_mode == 1:

            for belief in lambda_belief.belief_list:
                object_list = belief.obj_realization
                break

            for obj in object_list:
                current_entropy_list_per_object[obj][-1] = lambda_belief.entropy(obj,
                                                                                 entropy_lower_limit=entropy_lower_limit)
                current_MSDE_list_per_object[obj][-1] = lambda_belief.MSDE_obj(obj, class_GT[obj])
                entropy_collector += current_entropy_list_per_object[obj][-1]
                sigma_collector += 2 * np.sqrt(lambda_belief.lambda_covariance(obj)[0, 0]) / len(object_list)

        if Lambda_BSP_mode == 2:

            for obj in lambda_belief.object_lambda_dict:

                if use_bounds is False:

                    current_entropy_list_per_object[obj][-1] = lambda_belief.lambda_entropy_individual_numeric(
                        lambda_belief.object_lambda_dict[obj], entropy_lower_limit=entropy_lower_limit)

                else:

                    current_entropy_list_per_object[obj][-1], _ = lambda_belief.lambda_entropy_individual_bounds(
                        lambda_belief.object_lambda_dict[obj], entropy_lower_limit=entropy_lower_limit)

                current_MSDE_list_per_object[obj][-1] = lambda_belief.MSDE_obj(obj, class_GT[obj])
                entropy_collector += current_entropy_list_per_object[obj][-1]
                sigma_collector += 2 * np.sqrt(lambda_belief.
                                               lambda_covariance_numeric(
                    lambda_belief.object_lambda_dict[obj])[0, 0]) / len(lambda_belief.object_lambda_dict)

        current_entropy_list.append(entropy_collector)
        MSDE.append(lambda_belief.MSDE_expectation(class_GT))
        sigma_list.append(sigma_collector)

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
            width_offset = 0.0
            color_bar = 'blue'
    elif Lambda_BSP_mode == 1 and number_of_beliefs == 1:
        width_offset = 0.18
        color_bar = 'green'

    # if reward_mode == 1:
    #     color = 'r'
    # elif reward_mode == 2 and number_of_beliefs == 1:
    #     color = 'g'
    # elif reward_mode == 2 and number_of_beliefs != 1:
    #     color = 'b'

    # plotterdaac2d.GT_plotter_line(GT_poses, [], fig_num=0, show_plot=False,
    #                              plt_save_name=None, pause_time=None,
    #                              jpg_save=False, alpha=0.75, color=color_bar, ax=plt_ax)
    if Lambda_BSP_mode == 1:
        plt_ax = lambda_belief.belief_list[0].graph_realizations_in_one(idx + 1, fig_num=0, show_obj=True,
                                                                    show_weights=False, show_plot=False,
                                                                    show_elipses=True, plot_line=True, color=color_bar,
                                                                    single_plot=True, show_robot_pose=False, ax=plt_ax,
                                                                    enlarge=3)
    if Lambda_BSP_mode == 2:
        plt_ax = lambda_belief.display_graph(plot_line=True, display_title=False, show_plot=False, color=color_bar, ax=plt_ax,
                                             enlarge=3)

    # plt_ax.set_xlim(-8, 15)
    # plt_ax.set_ylim(-10, 10)
    plt_ax.plot(GT_pose[initial_image].x(), GT_pose[initial_image].y(), 'ro')
    plt_ax.text(GT_pose[initial_image].x(), GT_pose[initial_image].y(), 'Initial position')

    lambda_belief.lambda_bar_error_graph_multi(show_plt=False, ax=bar_ax, color=color_bar, width_offset=width_offset,
                                               GT_bar=True, GT_classes=class_GT)
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

    activate_MH = False
    number_of_seeds = 1
    number_of_beliefs = 1
    seed_offset = 293

    plt.rcParams.update({'font.size': 18})

    cls_model_1 = clsmodel_lg1.ClsModel('../exp_model_Book Jacket.pt',
                                               '../exp_model_Digital Clock.pt',
                                               '../exp_model_Packet.pt',
                                               '../exp_model_Pop Bottle.pt',
                                               '../exp_model_Soap Dispenser.pt',
                                               '../rinf_model_Book Jacket.pt',
                                               '../rinf_model_Digital Clock.pt',
                                               '../rinf_model_Packet.pt',
                                               '../rinf_model_Pop Bottle.pt',
                                               '../rinf_model_Soap Dispenser.pt')
    cls_model_2 = clsUmodel_fake_1d.JLPModel('../exp_model_Book Jacket.pt',
                                               '../exp_model_Digital Clock.pt',
                                               '../exp_model_Packet.pt',
                                               '../exp_model_Pop Bottle.pt',
                                               '../exp_model_Soap Dispenser.pt',
                                               '../rinf_model_Book Jacket.pt',
                                               '../rinf_model_Digital Clock.pt',
                                               '../rinf_model_Packet.pt',
                                               '../rinf_model_Pop Bottle.pt',
                                               '../rinf_model_Soap Dispenser.pt')
    random_object_flag = False

    fig_1, ax_red = plt.subplots()
    fig_2, ax_blue = plt.subplots()
    fig_3, ax_green = plt.subplots()
    fig_4, ax_bar_r1 = plt.subplots()

    fixed_stat = 'Exp'

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

    time_dict = {'JLP_R1': list(), 'MH_R1': list(), 'JLP_R2': list(), 'MH_R2': list(), 'WEU': list()}

    time_dict_track = {'JLP_R1': None, 'MH_R1': None, 'JLP_R2': None, 'MH_R2': None, 'WEU': None}

    for seed_idx in range(number_of_seeds):

        # Patten18arj like planning
        Lambda_BSP_mode = 1
        reward_mode = 2
        entropy, MSDE, _, time, time_dict_track['WEU'], _ = \
            motion_primitive_planning(seed_idx + seed_offset + 4, Lambda_BSP_mode, reward_mode, cls_model_1,
                                      number_of_beliefs=1, random_objects=random_object_flag,
                                      plt_ax=ax_green, bar_ax=ax_bar_r1)  # TODO: switch to r2 when needed.
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

        print('-O-O---------------------------SEED: ' + str(seed_idx) + ' --------------------------O-O-')

    ax_red.set_xlabel('X axis [m]')
    ax_red.set_ylabel('Y axis [m]')
    ax_red.set_xlim(-7, 7)
    ax_red.set_ylim(-7, 7)
    ax_blue.set_xlabel('X axis [m]')
    ax_blue.set_ylabel('Y axis [m]')
    ax_blue.set_xlim(-7, 7)
    ax_blue.set_ylim(-7, 7)
    ax_green.set_xlabel('X axis [m]')
    ax_green.set_ylabel('Y axis [m]')
    ax_green.set_xlim(-7, 7)
    ax_green.set_ylim(-7, 7)
    # ax_bar_r2.legend(['JLP R2', 'MH R2'])
    ax_bar_r1.legend(['WEU', 'JLP R1', 'JLP R2', 'MH R1', 'MH R2'], bbox_to_anchor=(0.5, 1), loc='lower center',
                     ncol=5)
    # ax_tick_override = ['Object 5', 'Object 14', 'Object 15', 'Object 29', 'Object 20', 'Object 3', 'Object 4', 'Object 19', 'Object 17', 'Object 23', 'Object 2']
    # ax_bar_r1.set_xticklabels(ax_tick_override)

    fig_1.tight_layout()
    fig_1.savefig('../figures/AVD_Black.eps')
    fig_2.tight_layout()
    fig_2.savefig('../figures/AVD_Blue.eps')
    fig_3.tight_layout()
    fig_3.savefig('../figures/AVD_Green.eps')
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
        # plt.title(r'Entropy of $b[\lambda]$ over time')
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        plt.legend(loc='lower left')
        fig.savefig('../figures/Plan_Exp_MH_R1.png')
        plt.show()

    # R2 comparison MH
    if activate_MH is True:
        fig = plt.figure(108)
        for idx in range(number_of_seeds):
            # plt.plot(JLP_entropy_R1[idx], color='black', alpha=1 / number_of_seeds, label='JLP')
            plt.plot(MH_e_entropy_R1[idx], color='purple', alpha=1 / number_of_seeds, label='MH R1')
            # plt.plot(JLP_entropy_R2[idx], color='blue', alpha=1 / number_of_seeds, label='JLP')
            plt.plot(MH_e_entropy_R2[idx], color='red', alpha=1 / number_of_seeds, label='MH R2')
        # plt.title(r'Entropy of $b[\lambda]$ over time')
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        plt.legend(loc='lower left')
        fig.savefig('../figures/Plan_Exp_MH_R2.png')
        plt.show()

    # R1 comparison JLP
    fig = plt.figure(101)
    for idx in range(number_of_seeds):
        plt.plot(JLP_entropy_R1[idx], color='black', alpha=1 / number_of_seeds, label='JLP R1')
        # plt.plot(-MH_entropy_R1[idx], color='purple', alpha=1 / number_of_seeds, label='MH R1')
        plt.plot(JLP_entropy_R2[idx], color='blue', alpha=1 / number_of_seeds, label='JLP R2')
        # plt.plot(-MH_entropy_R2[idx], color='red', alpha=1 / number_of_seeds, label='MH R2')
    # plt.title(r'Entropy of $b[\lambda]$ over time')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.tight_layout()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.legend(loc='lower left')
    fig.savefig('../figures/AVD_JLP_R1.eps')
    plt.show()

    # R2 comparison JLP
    fig = plt.figure(109)
    for idx in range(number_of_seeds):
        plt.plot(JLP_e_entropy_R1[idx], color='black', alpha=1 / number_of_seeds, label='JLP R1')
        # plt.plot(-MH_entropy_R1[idx], color='purple', alpha=1 / number_of_seeds, label='MH R1')
        plt.plot(JLP_e_entropy_R2[idx], color='blue', alpha=1 / number_of_seeds, label='JLP R2')
        # plt.plot(-MH_entropy_R2[idx], color='red', alpha=1 / number_of_seeds, label='MH R2')
    # plt.title(r'Entropy of $b[\lambda]$ over time')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.tight_layout()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.legend(loc='lower left')
    fig.savefig('../figures/Plan_Exp_JLP_R2.png')
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
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R1 - JLP_entropy_sigma_R1,
                     JLP_entropy_exp_R1 +
                     JLP_entropy_sigma_R1, color='black', alpha=0.2)
    # plt.plot(MH_entropy_exp_R1, color='purple', label='MH R1')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R1 - MH_entropy_sigma_R1, MH_entropy_exp_R1 +
    #                  MH_entropy_sigma_R1, color='purple', alpha=0.2)
    plt.plot(JLP_entropy_exp_R2, color='blue', label='JLP R2')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R2 - JLP_entropy_sigma_R2,
                     JLP_entropy_exp_R2 +
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
    # plt.title(r'Average entropy of $b[\lambda]$ over time')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.grid(True)
    plt.xlim(0, 20)
    fig.savefig('../figures/Plan_Stat_Exp_JLP_R1.png')
    plt.show()

    # Statistical R2 comparison JLP
    fig = plt.figure(1008)
    plt.plot(JLP_e_entropy_exp_R1, color='black', label='JLP R1')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_e_entropy_exp_R1 - JLP_e_entropy_sigma_R1,
                     JLP_e_entropy_exp_R1 +
                     JLP_e_entropy_sigma_R1, color='black', alpha=0.2)
    # plt.plot(MH_entropy_exp_R1, color='purple', label='MH R1')
    # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R1 - MH_entropy_sigma_R1, MH_entropy_exp_R1 +
    #                  MH_entropy_sigma_R1, color='purple', alpha=0.2)
    plt.plot(JLP_e_entropy_exp_R2, color='blue', label='JLP R2')
    plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_e_entropy_exp_R2 - JLP_e_entropy_sigma_R2,
                     JLP_e_entropy_exp_R2 +
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
    # plt.title(r'Average entropy of $b[\lambda]$ over time')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.grid(True)
    plt.xlim(0, 20)
    fig.savefig('../figures/Plan_Stat_Exp_JLP_R2.png')
    plt.show()

    # Statistical R1 comparison MH
    if activate_MH is True:
        fig = plt.figure(1002)
        # plt.plot(JLP_entropy_exp_R1, color='black', label='JLP R1')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R1 - JLP_entropy_sigma_R1, JLP_entropy_exp_R1 +
        #                  JLP_entropy_sigma_R1, color='black', alpha=0.2)
        plt.plot(MH_entropy_exp_R1, color='purple', label='MH R1')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R1 - MH_entropy_sigma_R1,
                         MH_entropy_exp_R1 +
                         MH_entropy_sigma_R1, color='purple', alpha=0.2)
        # plt.plot(JLP_entropy_exp_R2, color='blue', label='JLP R2')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R2 - JLP_entropy_sigma_R2, JLP_entropy_exp_R2 +
        #                  JLP_entropy_sigma_R2, color='blue', alpha=0.2)
        plt.plot(MH_entropy_exp_R2, color='red', label='MH R2')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R2 - MH_entropy_sigma_R2,
                         MH_entropy_exp_R2 +
                         MH_entropy_sigma_R2, color='red', alpha=0.2)
        # plt.plot(JLP_entropy_exp_R3, color='cyan', label='JLP R3')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R3 - JLP_entropy_sigma_R3, JLP_entropy_exp_R3 +
        #                  JLP_entropy_sigma_R3, color='cyan', alpha=0.2)
        # plt.plot(MH_entropy_exp_R3, color='orange', label='MH R3')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R3 - MH_entropy_sigma_R3, MH_entropy_exp_R3 +
        #                  MH_entropy_sigma_R2, color='orange', alpha=0.2)
        # plt.title(r'Average entropy of $b[\lambda]$ over time')
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_Stat_Exp_MH_R1.png')
        plt.show()

        # Statistical R2 comparison MH
        fig = plt.figure(1009)
        # plt.plot(JLP_entropy_exp_R1, color='black', label='JLP R1')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R1 - JLP_entropy_sigma_R1, JLP_entropy_exp_R1 +
        #                  JLP_entropy_sigma_R1, color='black', alpha=0.2)
        plt.plot(MH_e_entropy_exp_R1, color='purple', label='MH R1')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_e_entropy_exp_R1 - MH_e_entropy_sigma_R1,
                         MH_e_entropy_exp_R1 +
                         MH_e_entropy_sigma_R1, color='purple', alpha=0.2)
        # plt.plot(JLP_entropy_exp_R2, color='blue', label='JLP R2')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R2 - JLP_entropy_sigma_R2, JLP_entropy_exp_R2 +
        #                  JLP_entropy_sigma_R2, color='blue', alpha=0.2)
        plt.plot(MH_e_entropy_exp_R2, color='red', label='MH R2')
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_e_entropy_exp_R2 - MH_e_entropy_sigma_R2,
                         MH_e_entropy_exp_R2 +
                         MH_e_entropy_sigma_R2, color='red', alpha=0.2)
        # plt.plot(JLP_entropy_exp_R3, color='cyan', label='JLP R3')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_entropy_exp_R3 - JLP_entropy_sigma_R3, JLP_entropy_exp_R3 +
        #                  JLP_entropy_sigma_R3, color='cyan', alpha=0.2)
        # plt.plot(MH_entropy_exp_R3, color='orange', label='MH R3')
        # plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), MH_entropy_exp_R3 - MH_entropy_sigma_R3, MH_entropy_exp_R3 +
        #                  MH_entropy_sigma_R2, color='orange', alpha=0.2)
        # plt.title(r'Average entropy of $b[\lambda]$ over time')
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_Stat_Exp_MH_R2.png')
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
        # plt.title('Average MSDE over time')
        plt.xlabel('Time step')
        plt.ylabel('MSDE')
        plt.legend(loc='lower left')
        # plt.yscale('log')
        plt.tight_layout()
        plt.grid(True)
        # plt.ylim(0, 0.4)
        plt.xlim(0, 20)
        fig.savefig('../figures/AVD_MSDE.png')
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
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R1 - JLP_MSDE_sigma_R1,
                         JLP_MSDE_exp_R1 +
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
        # plt.title('Average MSDE over time')
        plt.xlabel('Time step')
        plt.ylabel('MSDE')
        plt.legend(loc='lower left')
        # plt.yscale('log')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_' + fixed_stat + '_MSDE_R1.png')
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
        plt.fill_between(np.array(range(len(JLP_entropy_exp_R1))), JLP_MSDE_exp_R2 - JLP_MSDE_sigma_R2,
                         JLP_MSDE_exp_R2 +
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
        # plt.title('Average MSDE over time')
        plt.xlabel('Time step')
        plt.ylabel('MSDE')
        plt.legend(loc='lower left')
        # plt.yscale('log')
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_' + fixed_stat + '_MSDE_R2.png')
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
    # plt.title('Average MSDE over time')
    plt.xlabel('Time step')
    plt.ylabel('MSDE')
    plt.legend(loc='lower left')
    # plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)
    # plt.ylim(0, 0.4)
    plt.xlim(0, 20)
    fig.savefig('../figures/AVD_MSDE_JLP.eps')
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
        # plt.title('Average MSDE over time')
        plt.xlabel('Time step')
        plt.ylabel('MSDE')
        plt.legend(loc='lower left')
        # plt.yscale('log')
        plt.tight_layout()
        plt.grid(True)
        # plt.ylim(0, 0.4)
        plt.xlim(0, 20)
        fig.savefig('../figures/Plan_Exp_MSDE_MH.png')
        plt.show()

    if activate_MH is True:
        fig = plt.figure(400)
        idx_time = 0
        key_color = {'JLP_R1': 'black', 'MH_R1': 'purple', 'JLP_R2': 'blue', 'MH_R2': 'red', 'WEU': 'green'}
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
    plt.text(len(time_dict_track['JLP_R2']) - 2, time_dict_track['JLP_R2'][-1] - 1, 'JLP')
    plt.plot(np.array(range(len(JLP_entropy_exp_R1))), time_dict_track['WEU'], color='green', label='JLP')
    plt.text(len(time_dict_track['WEU']) - 2, time_dict_track['WEU'][-1], 'WEU')
    # plt.title('Computation time of different methods with different number of beliefs')
    plt.xlabel('Time step')
    plt.ylabel('Time [s]')
    # plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(1, 20)
    fig.savefig('../figures/AVD_Time_Exp.eps')
    plt.show()


if __name__ == "__main__":
    main()
