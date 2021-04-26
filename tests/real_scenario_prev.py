from __future__ import print_function
import numpy as np
import scipy as sp
from scipy import special
from scipy import stats
import pandas as pd
import gtsam
import math
import damodel
import geomodel
import clsUmodel_real_2d as clsModel
import plotterdaac2d
import gaussianb_jlp
import matplotlib.pyplot as plt
from JLP_planner import JLPPLannerPrimitives
from hybridb import HybridBelief
from lambdab_lg import LambdaBelief
from lambda_planner import LambdaPlannerPrimitives
import time

# Initializations ----------------------------------------------------
sample_use_flag = True
ML_planning_flag = True
horizon = 1
ML_update = True
MCTS_flag = False
MCTS_branches = 3
cls_enable = True
display_graphs = True
measurement_radius_limit = 10
measurement_radius_minimum = 0
com_radius = 100000
class_GT = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
GT_cls_realization = ((1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1))
np.random.seed(120)
measurements_enabler = True
opening_angle = 180
opening_angle_rad = opening_angle * math.pi / 180
number_of_beliefs = 50
num_samp = 50
CVaR_flag = True
start_time = time.time()

# Set models ---------------------------------------------------------

np.set_printoptions(precision=3)
plt.rcParams.update({'font.size': 16})

prior_lambda = (1, 1, 1)
lambda_prior = np.array([special.digamma(prior_lambda[0: -1]) - special.digamma(prior_lambda[-1])])  #
lambda_prior = lambda_prior[0]
lambda_prior_noise_diag = np.array(special.polygamma(1, prior_lambda[0: -1]) + special.polygamma(1, prior_lambda[-1]))
lambda_prior_noise_matrix = np.diag(lambda_prior_noise_diag)
lambda_prior_noise = gtsam.noiseModel.Diagonal.Variances(lambda_prior_noise_diag)

da_model = damodel.DAModel(camera_fov_angle_horizontal=opening_angle_rad, range_limit=measurement_radius_limit,
                           range_minimum=measurement_radius_minimum)
geo_model_noise_diag = np.array([0.00001, 0.00001, 0.001, 0.01, 0.01, 0.001])
action_noise_diag = np.array([0.0003, 0.0003, 0.0001, 0.0003, 0.00030, 0.0001])
action_noise = gtsam.noiseModel.Diagonal.Variances(action_noise_diag)
geo_model_noise = gtsam.noiseModel.Diagonal.Variances(geo_model_noise_diag)
geo_model = geomodel.GeoModel(geo_model_noise)

cls_model = clsModel.JLPModel('../alexnet_exp_model_lg.pt', '../rand2_exp_model_lg.pt',
                              '../rand3_exp_model_lg.pt', '../alexnet_cov_model_lg.pt',
                              '../rand2_rinf_model_lg.pt', '../rand3_rinf_model_lg.pt')

# Quaternion flip check ----------------------------------------------

def create_quaternion_unflipped(w, qx, qy, qz):

    quat_first = gtsam.Rot3.Quaternion(w ,qx, qy, qz)
    flip_check = quat_first.rpy()*180/3.1415
    while flip_check[0] >= 180:
        flip_check[0] -= 360
    while flip_check[0] <= -180:
        flip_check[0] += 360
    if flip_check[0] < 135 and flip_check[0] > -135:
        return quat_first
    else:
        return gtsam.Rot3.Quaternion(qx, -w, -qz, qy)

# LOAD RAW DATA ------------------------------------------------------

chair_poses_load = pd.read_csv('../chair_poses.csv')
chair_poses = chair_poses_load.iloc[:, :]
chair_poses = chair_poses.values
robot_poses_load = pd.read_csv('../all_poses.csv')
robot_poses = robot_poses_load.iloc[:, :]
robot_poses = robot_poses.values
MC_dropout_lg_load = pd.read_csv('../mc_dropout_lg.csv')
MC_dropout_lg = MC_dropout_lg_load.iloc[:, :]
MC_dropout_lg = MC_dropout_lg.values
obs_angles_load = pd.read_csv('../obs_angle.csv')
obs_angles = obs_angles_load.iloc[:, :]
obs_angles = obs_angles.values
GT_DA_load = pd.read_csv('../gt_da.csv')
GT_DA = GT_DA_load.iloc[:, :]
GT_DA = GT_DA.values # GT_DA can be also used to index the measurements

chair_poses_angles = np.zeros([len(chair_poses), 4])
robot_poses_angles = np.zeros([len(robot_poses), 4])
ranges = np.zeros([len(MC_dropout_lg), 2])
actions = list()
GT_poses = list()
GT_objects = list()

# Generate measurements ----------------------------------------------------------

CONST45 = float(np.sqrt(2.0) / 2)

MC_dropout_lg_exp = MC_dropout_lg[:, 1:3]
MC_dropout_lg_cov = np.zeros([len(MC_dropout_lg), len(prior_lambda) - 1, len(prior_lambda) - 1])
for idx in range(len(MC_dropout_lg)):
    R_inf = np.matrix([[MC_dropout_lg[idx, 3], MC_dropout_lg[idx, 4]], [0.0, MC_dropout_lg[idx, 5]]])
    MC_dropout_lg_cov[idx, :, :] = np.linalg.inv(np.matmul(np.transpose(R_inf), R_inf))

for idx in range(len(chair_poses)):
    chair_poses_angles[idx][0] = idx

    GT_objects.append(gtsam.Pose3(create_quaternion_unflipped(chair_poses[idx][7] * CONST45 +
                                                              chair_poses[idx][6] * CONST45,
                                                            chair_poses[idx][4] * CONST45 -
                                                              chair_poses[idx][5] * CONST45,
                                                            chair_poses[idx][5] * CONST45 +
                                                              chair_poses[idx][4] * CONST45,
                                                            chair_poses[idx][6] * CONST45 -
                                                              chair_poses[idx][7] * CONST45),
                                      np.array([chair_poses[idx][1],
                                                   chair_poses[idx][2],
                                                   0.5])))
                                                   #chair_poses[idx][3])))

for idx in range(len(robot_poses)):
    robot_poses_angles[idx][0] = idx
    #print(robot_poses[idx][4:])
    #print(temp)

    if idx is not 0:
        robot_pose_1 = gtsam.Pose3(create_quaternion_unflipped(robot_poses[idx][7],
                                                         robot_poses[idx][4],
                                                         robot_poses[idx][5],
                                                         robot_poses[idx][6]),
                                   np.array([robot_poses[idx][1],
                                                robot_poses[idx][2],
                                                robot_poses[idx][3]-0.7]))
        robot_pose_2 = gtsam.Pose3(create_quaternion_unflipped(robot_poses[idx - 1][7],
                                                         robot_poses[idx - 1][4],
                                                         robot_poses[idx - 1][5],
                                                         robot_poses[idx - 1][6]),
                                   np.array([robot_poses[idx - 1][1],
                                                robot_poses[idx - 1][2],
                                                robot_poses[idx - 1][3]-0.7]))
        if idx is 1:
            GT_poses.append(robot_pose_2)
        GT_poses.append(robot_pose_1)
        actions.append(robot_pose_2.between(robot_pose_1))

geo_measurement_dict = dict()
sem_measurement_exp_dict = dict()
sem_measurement_cov_dict = dict()
expected_sem_measurement_dict = dict()
DA_dict = dict()

for idx in range(len(MC_dropout_lg)):
    if GT_DA[idx][1] <= len(GT_objects):
        geo_measurement_dict[GT_DA[idx][0]] = GT_poses[GT_DA[idx][0]].between(GT_objects[GT_DA[idx][1] - 1])
        sem_measurement_exp_dict[GT_DA[idx][0]] = MC_dropout_lg_exp[idx]
        sem_measurement_cov_dict[GT_DA[idx][0]] = MC_dropout_lg_cov[idx]
        DA_dict[GT_DA[idx][0]] = GT_DA[idx, 1]
        geo_cov = np.diag(geo_model_noise_diag)

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

# Plot prior lines

fig = plt.figure(0)
ax = fig.gca()

plotterdaac2d.GT_plotter_line(GT_poses, GT_objects,
                              limitx=[-10, 7], limity=[-8, 8], show_plot=False, ax=ax,
                              start_pose=GT_poses[0])
ax.set_xlabel('X axis [m]')
ax.set_ylabel('Y axis [m]')
plt.tight_layout()
plt.show()

# Belief initialization --------------------------------------------------

prior_noise_diag = np.array([00.000001, 0.000001, 0.0000038, 00.00002, 00.0000202, 0.000001])
prior_noise = gtsam.noiseModel.Diagonal.Variances(prior_noise_diag)
prior_noise_cov = np.diag(prior_noise_diag)

JLP_belief = gaussianb_jlp.JLPBelief(geo_model, da_model, cls_model, GT_poses[0], prior_noise,
                                     lambda_prior_mean=lambda_prior,
                                     lambda_prior_noise=lambda_prior_noise_matrix,
                                     cls_enable=cls_enable)

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

# Arrange the measurements

geo_measurement_divided = dict()
sem_measurement_exp_divided = dict()
sem_measurement_cov_divided = dict()

for obs_idx in geo_measurement_dict:
    geo_part = geo_measurement_dict[obs_idx]
    geo_cov = np.diag(geo_model_noise_diag)
    geo_measurement_divided[obs_idx] = geo_measurement_generator(geo_part, geo_cov, no_noise=False)
    sem_measurement_exp_divided[obs_idx] = sem_measurement_exp_dict[obs_idx]
    sem_measurement_cov_divided[obs_idx] = sem_measurement_cov_dict[obs_idx]

# Inference

#for idx in range(len(GT_poses) - 1):
for idx in range(10):

    print('Time step k=' + str(idx + 1) + '\n---------------------------------------')

    if actions[idx]:
        action_generated = action_generator(actions[idx], action_noise_diag, no_noise=False)
        JLP_belief.action_step(actions[idx], action_noise)

    geo_measurement = list()
    sem_measurement_exp = list()
    sem_measurement_cov = list()
    GT_DA_single_step = list()

    for jdx in geo_measurement_divided:
        if idx == jdx:

            geo_measurement.append(geo_measurement_divided[jdx])
            sem_measurement_exp.append(sem_measurement_exp_divided[jdx])
            sem_measurement_cov.append(sem_measurement_cov_divided[jdx])
            GT_DA_single_step.append(DA_dict[jdx])

    JLP_belief.add_measurements(geo_measurement, sem_measurement_exp, sem_measurement_cov,
                                GT_DA_single_step)

JLP_belief.print_results(show_entropy=False)
#JLP_belief.simplex_3class(1, show_plot=True, log_likelihood=False)
#JLP_belief.display_graph()