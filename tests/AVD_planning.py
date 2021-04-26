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
from PIL import Image

# Mode choosing
Lambda_BSP_mode = 2  # 1: Multi-Hybrid. 2: JLP.
reward_mode = 1 #1: Entropy #2: Expectation entropy #3: Information Gain
inf_planning_mode = 2 #1: Inference #2: Planning
use_bounds = True


## --------------- PRLIMINARIES: NETWORK AND SCENARIO SETUP

# Find the corresponding depth image to the jpg
def depth_image_find(image_address):

    depth_address = image_address[0:-5]
    depth_address += '3.png'
    return depth_address

# Load json file
actions = ['forward','rotate_ccw','rotate_cw','backward','left','right']

scenario_file = open('../../../Office_001_1/annotations.json')
scenario_data = json.load(scenario_file)

classification_file = open('../classification_data.json')
classification_data = json.load(classification_file)

#------------------------------------- EXTRACT GROUND TRUTH

detailed_GT_file = scipy.io.loadmat('../../../Office_001_1/image_structs.mat')

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
sample_use_flag = False
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
np.random.seed(300)
measurements_enabler = True
opening_angle = 50
opening_angle_rad = opening_angle * math.pi / 180
number_of_beliefs = 20
num_samp = 100
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
if Lambda_BSP_mode == 1:#TODO: PROGRAM IT IN AFTER JLP WORKS
    import clsmodel_real_lg4 as clsmodel_lg1
    cls_model = clsmodel_lg1.ClsModel('../exp_model_Book Jacket.pt',
                                           '../exp_model_Digital Clock.pt',
                                           '../exp_model_Packet.pt',
                                           '../exp_model_Pop Bottle.pt',
                                           '../exp_model_Soap Dispenser.pt',
                                           '../rinf_model_Book Jacket.pt',
                                           '../rinf_model_Digital Clock.pt',
                                           '../rinf_model_Packet.pt',
                                           '../rinf_model_Pop Bottle.pt',
                                           '../rinf_model_Soap Dispenser.pt')
if Lambda_BSP_mode == 2:
    import clsUmodel_real_4d as clsUmodel_fake_1d
    cls_model = clsUmodel_fake_1d.JLPModel('../exp_model_Book Jacket.pt',
                                           '../exp_model_Digital Clock.pt',
                                           '../exp_model_Packet.pt',
                                           '../exp_model_Pop Bottle.pt',
                                           '../exp_model_Soap Dispenser.pt',
                                           '../rinf_model_Book Jacket.pt',
                                           '../rinf_model_Digital Clock.pt',
                                           '../rinf_model_Packet.pt',
                                           '../rinf_model_Pop Bottle.pt',
                                           '../rinf_model_Soap Dispenser.pt')



# Robot poses GT
initial_image = '100110000350101.jpg'
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
                         cls_enable=cls_enable, pruning_threshold=1000, ML_update=ML_update))

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

reward_collections = list()
chosen_action_list = list()
current_entropy_list = list()
current_entropy_list.append(0)
MSDE = list()
MSDE.append(4/25)

class_GT = {4: 3, 24: 5, 10: 2, 1: 2, 29: 2, 15: 2, 20: 2, 31: 2, 5: 3, 16: 2}

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

    if inf_planning_mode == 2:
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
    lambda_belief.action_step(action, action_noise)
    current_image = scenario_data[current_image][action_name]
    print(current_image)

    # Create the ground truth data association
    GT_DA = list()
    geo_measurements = list()
    sem_measurements = list()
    sem_measurements_exp = list()
    sem_measurements_cov = list()
    idx_range = 0
    for angle in classification_data[current_image]['Angles']:
        if classification_data[current_image]['Range'][idx_range] != 0: # and \
                #classification_data[current_image]['DA'][idx_range][0] != 10:  # TODO:SECOND TEMPORARY CONDITION
            GT_DA.append(classification_data[current_image]['DA'][idx_range][0])
            # geo_measurements.append([angle[0], angle[1], classification_data[current_image]['Range'][idx_range]])
            geo_measurements.append([angle[0], -0.0, classification_data[current_image]['Range'][idx_range]])

            if Lambda_BSP_mode == 1:
                sem_measurements.append(classification_data[current_image]['CPV'][
                                            idx_range])  # TODO: If MH needs more than 1 measurement change it

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

    if Lambda_BSP_mode == 1:
        ax = lambda_belief.belief_list[0].graph_realizations_in_one(idx + 1, fig_num=0, show_obj=True,
                                                                    show_weights=False, show_plot=False,
                                                                    show_elipses=False)
    #plt.show()

print(chosen_action_list)

name = 'plan_10'

if Lambda_BSP_mode == 1:
    ax = lambda_belief.belief_list[0].graph_realizations_in_one(idx + 1, fig_num=0, show_obj=True,
                                                              show_weights=False, show_plot=False, show_elipses=False)
if Lambda_BSP_mode == 2:
    ax = lambda_belief.display_graph(plot_line=True, display_title=False, show_plot=False)

if inf_planning_mode == 1:
    passive_active = 'Inf'
elif inf_planning_mode == 2:
    passive_active = 'Plan'

if Lambda_BSP_mode is 1:
    mh_jlp = 'MH'
elif Lambda_BSP_mode is 2:
    mh_jlp = 'JLP'

if inf_planning_mode == 2:
    if reward_mode == 1:
        passive_active += '_R1'
    elif reward_mode == 2:
        passive_active += '_R2'

# plotterdaac2d.GT_plotter_line(GT_poses, [], fig_num=0, ax=ax, show_plot=True,
#                               plt_save_name=None, pause_time=None, limitx=[-2, 15], limity=[-8, 8],
#                               red_point=idx + 1, jpg_save=False)

fig = plt.figure(111)
print(current_entropy_list)
plt.plot(range(track_length + 1), current_entropy_list)
#plt.title('Entropy over time')
plt.xlabel('Time step')
plt.ylabel('Entropy')
plt.xlim(0, track_length)
plt.grid(True)
plt.tight_layout()
plt.show()

fig = plt.figure(222)
print(current_entropy_list)
for obj in class_GT:
    plt.plot(range(track_length + 1), current_entropy_list_per_object[obj], alpha=2 / np.sqrt(len(class_GT)))
    plt.text(track_length - 1, current_entropy_list_per_object[obj][-1], 'O' + str(obj))
#plt.title('Entropy over time per object')
plt.xlabel('Time step')
plt.ylabel('Entropy')
plt.xlim(0, track_length)
plt.grid(True)
plt.tight_layout()
plt.show()

fig_2 = plt.figure(333)
plt.plot(range(track_length + 1), MSDE)
#plt.title('Classification accuracy MSDE over time')
plt.xlabel('Time step')
plt.ylabel('MSDE')
#plt.yscale('log')
plt.ylim(0, 1)
plt.xlim(0, track_length)
plt.grid(True)
plt.tight_layout()
if Lambda_BSP_mode is 1 and number_of_beliefs > 1:
    plt.savefig('../figures/AVD_MSDE_MH' + str(number_of_beliefs) + '.png')
elif Lambda_BSP_mode is 1 and number_of_beliefs == 1:
    plt.savefig('../figures/AVD_MSDE_WEU.png')
elif Lambda_BSP_mode is 2:
    plt.savefig('../figures/AVD_MSDE_JLP.png')
plt.show()

fig = plt.figure(334)
print(current_entropy_list)
for obj in class_GT:
    plt.plot(range(track_length + 1), current_MSDE_list_per_object[obj], alpha=2 / np.sqrt(len(class_GT)))
    plt.text(track_length - 1, current_MSDE_list_per_object[obj][-1], 'O' + str(obj))
#plt.title('MSDE over time per object')
plt.xlabel('Time step')
plt.ylabel('MSDE')
#plt.yscale('log')
plt.ylim(0, 1)
plt.xlim(0, track_length)
plt.grid(True)
#plt.tight_layout()
if Lambda_BSP_mode is 1 and number_of_beliefs > 1:
    plt.savefig('../figures/AVD_MSDE_MH' + str(number_of_beliefs) + '_ind.png')
elif Lambda_BSP_mode is 1 and number_of_beliefs == 1:
    plt.savefig('../figures/AVD_MSDE_WEU_ind.png')
elif Lambda_BSP_mode is 2:
    plt.savefig('../figures/AVD_MSDE_JLP_ind.png')
plt.show()

fig_2 = plt.figure(444)
plt.plot(range(track_length), sigma_list)
plt.title('Sigma width over time')
plt.xlabel('Time step')
plt.ylabel('Sigma')
plt.grid(True)
plt.tight_layout()
plt.show()
