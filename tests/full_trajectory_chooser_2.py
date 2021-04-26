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

# Mode choosing
Lambda_BSP_mode = 2  # 1: Multi-Hybrid. 2: JLP.

# Initializations ----------------------------------------------------
sample_use_flag = True
ML_planning_flag = True
horizon = 1
ML_update = True
MCTS_flag = False
MCTS_branches = 3
cls_enable = True
display_graphs = True
measurement_radius_limit = 25.5
measurement_radius_minimum = 0
com_radius = 100000
class_GT = {1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1, 8: 1, 9: 2, 10: 1}
GT_cls_realization = ((1, 1), (2, 2), (3, 1), (4, 1), (5, 1), (6, 2), (7, 1), (8, 1))
np.random.seed(15)
measurements_enabler = True
opening_angle = 180
opening_angle_rad = opening_angle * math.pi / 180
number_of_beliefs = 2
num_samp = 50
CVaR_flag = True
start_time = time.time()

# Set models ---------------------------------------------------------

np.set_printoptions(precision=3)
plt.rcParams.update({'font.size': 16})

prior_lambda = (1, 1)
lambda_prior = np.array([special.digamma(prior_lambda[0]) - special.digamma(prior_lambda[1])]) #
lambda_prior_noise_diag = np.matrix(special.polygamma(1, prior_lambda[0]) + special.polygamma(1, prior_lambda[1]))
lambda_prior_noise = gtsam.noiseModel.Diagonal.Variances(lambda_prior_noise_diag)

da_model = damodel.DAModel(camera_fov_angle_horizontal=opening_angle_rad, range_limit=measurement_radius_limit,
                           range_minimum=measurement_radius_minimum)
geo_model_noise_diag = np.array([0.00001, 0.00001, 0.001, 0.01, 0.01, 0.001])
geo_model_noise = gtsam.noiseModel.Diagonal.Variances(geo_model_noise_diag)
geo_model = geomodel_proj.GeoModel(geo_model_noise)
if Lambda_BSP_mode == 1:
    import clsmodel_lg1
    cls_model = clsmodel_lg1.ClsModel()
if Lambda_BSP_mode == 2:
    import clsUmodel_fake_1d
    cls_model = clsUmodel_fake_1d.JLPModel()
# DEFINE GROUND TRUTH -------------------------------------------------

# Objects
GT_objects = list()
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, -2.0, 0.0])))
# GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi/2, 0.0, 0.0), np.array([1.0, 2.0, 0.0])))
# GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi/2, 0.0, 0.0), np.array([-2.5, 3.5, 0.0])))
# GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3*math.pi/4, 0.0, 0.0), np.array([1.0, 4.0, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([-6.0, -3.0, 0.0])))
# GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi/4, 0.0, 0.0), np.array([3.0, -2.0, 0.0])))
# GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3*math.pi/2, 0.0, 0.0), np.array([1.5,-6.0, 0.0])))
# GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi, 0.0, 0.0), np.array([-1.0, -4.0, 0.0])))
# GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi, 0.0, 0.0 / 2), np.array([-1.5, -1.0, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([-6.5, 1.0, 0.0])))

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
GT_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(-2*math.pi/2, 0.0, 0.0), np.array([0.0, 0.0, 0.0])))

# DEFINE ACTION SEQUENCES ----------------------------------------------

sequence = list()
sequence_length = 20

# Command sequence generation
for idx_seq in range(len(Init_poses)):

    sequence.append(list())
    sequence[-1].append(gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))

    for idx in range(sequence_length):
        sequence[-1].append(gtsam.Pose3(gtsam.Pose2(0.7, 0.0, np.random.normal(0, 0.3))))
        #sequence[-1].append(gtsam.Pose3(gtsam.Pose2(0.25, 0.0, 0.0)))

#------------------------------------------------ PRIORS

# Robot priors
#prior = gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0]))
prior = GT_poses[0]
prior_noise_diag = np.array([00.000001, 0.000001, 0.0000038, 00.00002, 00.0000202, 0.000001])
prior_noise = gtsam.noiseModel.Diagonal.Variances(prior_noise_diag)
prior_noise_cov = np.diag(prior_noise_diag)

lambda_belief_list = list()
lambda_planner_list = list()

# Create belief based on the switch
if Lambda_BSP_mode == 1:

    belief_list = list()
    lambda_planner_list = list()

    for idx in range(number_of_beliefs * len(sequence)):
        # cls_prior_rand_0 = np.random.dirichlet(prior_lambda)
        cls_prior_rand_0_lg = np.random.multivariate_normal(lambda_prior, lambda_prior_noise_diag)
        cls_prior_rand_0_con = np.exp(cls_prior_rand_0_lg) / (1 + np.sum(np.exp(cls_prior_rand_0_lg)))
        cls_prior_rand_0 = np.concatenate((cls_prior_rand_0_con, 1 /
                                           (1 + np.sum(np.exp(cls_prior_rand_0_lg)))), axis=None)
        cls_prior_rand = cls_prior_rand_0.tolist()


        belief_list.append(
            HybridBelief(2, geo_model, da_model, cls_model, cls_prior_rand, Init_poses[idx // number_of_beliefs],
                         prior_noise,
                         cls_enable=cls_enable, pruning_threshold=7, ML_update=ML_update))

    for idx_1 in range(len(sequence)):
        lambda_belief_list.append(LambdaBelief(belief_list[number_of_beliefs * idx_1: number_of_beliefs * (idx_1 + 1)]))
        lambda_planner_list.append(LambdaPlannerPrimitives(lambda_belief_list[-1]))

if Lambda_BSP_mode == 2:
    for idx_1 in range(len(sequence)):
        lambda_belief_list.append(gaussianb_jlp.JLPBelief(geo_model, da_model, cls_model, Init_poses[idx_1], prior_noise,
                                                          lambda_prior_mean=lambda_prior,
                                                          lambda_prior_noise=lambda_prior_noise_diag,
                                                          cls_enable=cls_enable))
        lambda_planner_list.append(JLPPLannerPrimitives(lambda_belief_list[-1]))

# Plot all the GT sequences
fig = plt.figure(0)
ax = fig.gca()
for sequence_idx in range(len(sequence)):

    GT_sequence = list()
    #GT_sequence.append(GT_poses[0])
    GT_sequence.append(Init_poses[sequence_idx])

    for action_idx in range(len(sequence[sequence_idx])):
        GT_sequence.append(GT_sequence[-1].compose(sequence[sequence_idx][action_idx]))

    plotterdaac2d.GT_plotter_line(GT_sequence, GT_objects, color=(0.8, 0.4, 0.4),
                                  limitx=[-10, 7], limity=[-8, 8], show_plot=False, ax=ax,
                                  start_pose=GT_sequence[-1], robot_id=str(sequence_idx + 1))
ax.set_xlabel('X axis [m]')
ax.set_ylabel('Y axis [m]')
plt.tight_layout()
plt.show()

# SEMANTIC MEASUREMENT GENERATION FUNCTION-----------------------------------------------------

def sem_measurements_generator(relative_pose, no_noise=False):

    radius = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2)
    psi = np.arctan2(-relative_pose.y(), -relative_pose.x())
    theta = np.arctan2(-relative_pose.z(), radius)

    K = 2.75
    Alpha = 0.25

    #Sigma = 1 / K
    Sigma = 1 / (K * (0.6 + 0.4 * np.cos(psi + theta)))

    #f = Alpha * np.sin(psi + theta) + (1 - Alpha)
    f = 0.75

    sem_gen = -1

    if no_noise is False:
        while sem_gen < 0 or sem_gen > 1:
            sem_gen = np.random.normal(f,Sigma)
    else:
        sem_gen = f

    sem_gen_vector = [ sem_gen , 1 - sem_gen]

    return sem_gen_vector


def sem_measurements_generator_2(relative_pose, no_noise=False):

    radius = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2)
    psi = np.arctan2(-relative_pose.y(), -relative_pose.x())
    theta = np.arctan2(-relative_pose.z(), radius)

    K = 2.75
    Alpha = 0.25

    #Sigma = 1 / K
    Sigma = 1 / (K * (0.6 + 0.4 * np.cos(psi + theta)))

    #f = - Alpha * np.sin(psi + theta) + Alpha
    f = 0.25

    sem_gen = -1

    if no_noise is False:
        while sem_gen < 0 or sem_gen > 1:
            sem_gen = np.random.normal(f,Sigma)
    else:
        sem_gen = f

    sem_gen_vector = [ sem_gen , 1 - sem_gen]

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
action_noise_diag = np.array([0.0003, 0.0003, 0.0001, 0.03, 0.030, 0.001])
action_noise = gtsam.noiseModel.Diagonal.Variances(action_noise_diag)

reward_collections = list()

for sequence_idx in range(len(sequence)):

    # zero_action = gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0))
    lambda_belief_list[sequence_idx].action_step(sequence[sequence_idx][0], action_noise)

    GT_DA = list()
    geo_measurements = list()
    sem_measurements_exp = list()
    sem_measurements_cov = list()
    if Lambda_BSP_mode == 1:
        sem_measurements = list()
        for lambda_real in range(len(belief_list)):
            sem_measurements.append(list())


    for obj_idx in range(len(GT_objects)):

        geo_part = Init_poses[sequence_idx].between(GT_objects[obj_idx])
        #geo_part = GT_poses[0].between(GT_objects[obj_idx])
        # Check if the object is seen
        xy_angle = np.arctan2(GT_objects[obj_idx].y() - Init_poses[sequence_idx].y(),
                              GT_objects[obj_idx].x() - Init_poses[sequence_idx].x())
        angles = Init_poses[sequence_idx].rotation().matrix()
        psi = np.arctan2(angles[1, 0], angles[0, 0])

        geo_radius = np.sqrt(geo_part.x() ** 2 + geo_part.y() ** 2 + geo_part.z() ** 2)

        if np.abs(xy_angle - psi) <= opening_angle_rad and geo_radius <= measurement_radius_limit:

            GT_DA.append(obj_idx + 1)

            geo_cov = np.diag(geo_model_noise_diag)
            meas_relative_pose = geo_measurement_generator(geo_part, geo_cov, no_noise=True)

            geo_measurements.append(meas_relative_pose)

            #-------------------------------###
            if Lambda_BSP_mode == 1:

                for lambda_real in range(len(belief_list)):

                    if class_GT[obj_idx + 1] == 2:
                        sem_part = sem_measurements_generator_2(GT_objects[obj_idx].between(GT_poses[0]),
                                                                no_noise=False)
                        # print('Object: ' + str(j + 1) + ',sem: ' + str(sem_part) + ',robot ' + robot_id + ',Class 2')
                        sem_part = [0.5, 0.5]
                        # sem_part[0] = np.random.uniform(0, 1)
                        # sem_part[1] = 1 - sem_part[0]
                    else:
                        sem_part = sem_measurements_generator(GT_objects[obj_idx].between(GT_poses[0]),
                                                              no_noise=False)
                        sem_part = [0.5, 0.5]
                        # sem_part[0] = np.random.uniform(0, 1)
                        # sem_part[1] = 1 - sem_part[0]

                    sem_measurements[lambda_real].append(sem_part)  # GENERATED

            # -------------------------------###
            if Lambda_BSP_mode == 2:
                sem_measurements_exp.append([0.])
                sem_measurements_cov.append(np.array([[0.000005]]))


    lambda_plots_flag = False
    # if sequence_idx in [0, 1, 4, 8, 12, 15]:
    #     lambda_plots_flag = True
    # else:
    #     lambda_plots_flag = False

    # -------------------------------###
    if Lambda_BSP_mode == 1:
        lambda_belief_list[sequence_idx].add_measurements(geo_measurements, sem_measurements, da_current_step=GT_DA,
                                                          number_of_samples=num_samp, new_input_object_prior=None,
                                                          new_input_object_covariance=None)

        reward_list = lambda_planner_list[sequence_idx].evaluate_trajectory(sequence[sequence_idx][1:], action_noise,
                                                                            geo_model_noise, ML_planning=False,
                                                                            return_sub_costs=True,
                                                                            lambda_plots=lambda_plots_flag,
                                                                            CVaR_flag=CVaR_flag)

    # -------------------------------###
    if Lambda_BSP_mode == 2:
        lambda_belief_list[sequence_idx].add_measurements(geo_measurements, sem_measurements_exp, sem_measurements_cov,
                                                          GT_DA)
        reward_list = lambda_planner_list[sequence_idx].evaluate_trajectory(sequence[sequence_idx][1:], action_noise,
                                                                            np.diag(geo_model_noise_diag),
                                                                            ML_planning=ML_planning_flag,
                                                                            return_sub_costs=True,
                                                                            lambda_plots=lambda_plots_flag,
                                                                            CVaR_flag=CVaR_flag,
                                                                            number_of_samples=number_of_beliefs,
                                                                            sample_use_flag=sample_use_flag)


    print('Sequence number: ' + str(sequence_idx))
    reward_collections.append(reward_list)

# Print running time
print(time.time() - start_time)

# Prepare to present an action sequence to reward graph
x_array = np.zeros([len(sequence) * number_of_beliefs])
x_array_reduced = np.zeros([len(sequence)])
y_array = np.zeros([len(sequence) * number_of_beliefs])
y_array_reduced = np.zeros([len(sequence)])
y_array_reduced_exp = np.zeros([len(sequence)])
for sequence_idx in range(len(sequence)):
    x_array_reduced[sequence_idx] = sequence_idx + 1
    for belief_idx in range(number_of_beliefs):
        x_array[sequence_idx * number_of_beliefs + belief_idx] = sequence_idx + 1
        #---------##
        if ML_planning_flag is False:
            y_array[sequence_idx * number_of_beliefs + belief_idx] = reward_collections[sequence_idx][belief_idx]
        else:
            y_array[sequence_idx * number_of_beliefs + belief_idx] = reward_collections[sequence_idx]

    if CVaR_flag is True and ML_planning_flag is False:
        CVaR_threshold = 0.25
        sub_costs_np = np.sort(y_array[sequence_idx * number_of_beliefs: (sequence_idx + 1) * number_of_beliefs])
        last_index = int(CVaR_threshold * len(sequence))
        sub_costs_cut = sub_costs_np[0:last_index + 1]
        CVaR_cost = np.average(sub_costs_cut)
        y_array_reduced[sequence_idx] = CVaR_cost
    else:
        y_array_reduced[sequence_idx] = np.average(y_array[sequence_idx * number_of_beliefs:
                                                           (sequence_idx + 1) * number_of_beliefs])

    y_array_reduced_exp[sequence_idx] += prior_entropy

# print(reward_collections)
plt.grid(True)
plt.xlabel('Action index')
plt.ylabel('-H(lambda)')
plt.plot(x_array, y_array, 'o', color='black')
plt.plot(x_array_reduced, y_array_reduced, 'o', color='red')
plt.plot(x_array_reduced, -y_array_reduced_exp, 'o', color='blue')
# plt.yscale('log')
# plt.axis([0, 16, 0, 10])
plt.show()

print('highest objective function value: ' + str(y_array_reduced[0]))

# Create graph
