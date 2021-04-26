from __future__ import print_function
import numpy as np
import gtsam
import math
import damodel
import geomodel as geomodel_proj
import clsmodel_lg1 as clsmodel
import plotterdaac2d
import matplotlib.pyplot as plt
from hybridb import HybridBelief
from lambdab import LambdaBelief
import time as time_counter

# Initializations ----------------------------------------------------
ML_update = False
num_samp = 400
cls_enable = True
display_graphs = False
measurement_radius_limit = 7
com_radius = 10
class_GT = {1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1, 8: 1}
GT_cls_realization = ((1,1),(2,2),(3,1),(4,1),(5,1),(6,2),(7,1),(8,1))
np.random.seed(3)
measurements_enabler = True

# Set models ---------------------------------------------------------

np.set_printoptions(precision=3)
plt.rcParams.update({'font.size': 16})

cls_prior = np.array([0.5, 0.5])
da_model = damodel.DAModel()
geo_model_noise_diag = np.array([0.00001, 0.00001, 0.01, 0.1, 0.1, 0.00001])
geo_model_noise = gtsam.noiseModel.Diagonal.Variances(geo_model_noise_diag)
geo_model = geomodel_proj.GeoModel(geo_model_noise)
cls_model = clsmodel.ClsModel()

# DEFINE GROUND TRUTH -------------------------------------------------

# Objects
GT_objects = list()
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(3*math.pi/8, 0.0, 0.0), np.array([0.0, -2.0, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi/2, 0.0, 0.0), np.array([1.0, 2.0, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi/2, 0.0, 0.0), np.array([-2.5, 3.5, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3*math.pi/4, 0.0, 0.0), np.array([1.0, 4.0, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi/2, 0.0, 0.0), np.array([-6.0, -3.0, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi/4, 0.0, 0.0), np.array([-4.0, -2.0, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3*math.pi/2, 0.0, 0.0), np.array([1.5,-6.0, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([-1.0, -4.0, 0.0])))

# Action list
actions = list()
action_noise_diag = np.array([0.0003, 0.0003, 0.01, 0.03, 0.030, 0.001])
action_noise = gtsam.noiseModel.Diagonal.Variances(action_noise_diag)

actions.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.04, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
actions.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0.04, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
actions.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.08, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
actions.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0.08, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))

# Robot poses
GT_poses = list()
GT_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi/2, 0.0, 0.0), np.array([2.0, 8.0, 0.0])))
for idx in range(60):

    GT_poses.append(GT_poses[idx].compose(actions[1]))

    if idx >= 4:
        del GT_poses[-1]
        GT_poses.append(GT_poses[idx].compose(actions[3]))
    if idx >= 21:
        del GT_poses[-1]
        GT_poses.append(GT_poses[idx].compose(actions[1]))
    if idx >= 24:
        del GT_poses[-1]
        GT_poses.append(GT_poses[idx].compose(actions[2]))

fig = plt.figure(0)
ax = fig.gca()
color = (0.8, 0.4, 0.4)

plotterdaac2d.GT_plotter_line(GT_poses, GT_objects, color=color,
                              limitx=[-10, 7], limity=[-8, 8], show_plot=False, ax=ax,
                              start_pose=GT_poses[0], robot_id='R')
ax.set_xlabel('X axis [m]')
ax.set_ylabel('Y axis [m]')
plt.tight_layout()
plt.show()

# DEFINE PRIORS -----------------------------------------------------------------------

# Object priors
object_prior = list()
object_prior.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])))
object_prior.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])))
object_prior.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])))
object_prior.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])))
object_prior.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])))
object_prior.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])))
object_prior.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])))
object_prior.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])))
object_prior_noise_diag = np.array([5000.05, 5000.05, 5000.0005, 5000.05, 5000.05, 5000.05])
object_prior_noise = gtsam.noiseModel.Diagonal.Variances(object_prior_noise_diag)

# Robot priors
prior = gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi/2, 0.0, 0.0), np.array([2.0, 8.0, 0.0]))
prior_noise_diag = np.array([00.000001, 0.000001, 0.0000038, 00.00002, 00.0000202, 0.000001])
prior_noise = gtsam.noiseModel.Diagonal.Variances(prior_noise_diag)

# Noise cov
prior_noise_cov = np.diag(prior_noise_diag)
object_prior_noise_cov = np.diag(object_prior_noise_diag)

# Belief initializations
belief_list = list()
belief_list.append(HybridBelief(2, geo_model, da_model, cls_model, cls_prior, prior, prior_noise, cls_enable=cls_enable,
                                pruning_threshold=7, ML_update=ML_update))
belief_list.append(HybridBelief(2, geo_model, da_model, cls_model, cls_prior, prior, prior_noise, cls_enable=cls_enable,
                                pruning_threshold=7, ML_update=ML_update))
belief_list.append(HybridBelief(2, geo_model, da_model, cls_model, cls_prior, prior, prior_noise, cls_enable=cls_enable,
                                pruning_threshold=7, ML_update=ML_update))
belief_list.append(HybridBelief(2, geo_model, da_model, cls_model, cls_prior, prior, prior_noise, cls_enable=cls_enable,
                                pruning_threshold=7, ML_update=ML_update))
belief_list.append(HybridBelief(2, geo_model, da_model, cls_model, cls_prior, prior, prior_noise, cls_enable=cls_enable,
                                pruning_threshold=7, ML_update=ML_update))
lambda_belief = LambdaBelief(belief_list)


# SEMANTIC MEASUREMENT GENERATION FUNCTION-----------------------------------------------------

def sem_measurements_generator(relative_pose, no_noise=False):

    radius = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2)
    psi = np.arctan2(-relative_pose.y(), -relative_pose.x())
    theta = np.arctan2(-relative_pose.z(), radius)

    K = 2
    Alpha = 0.35

    Sigma = 1 / K

    f = Alpha * np.sin(psi + theta) + (1 - Alpha)

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

    K = 2
    Alpha = 0.35

    Sigma = 1 / K

    f = - Alpha * np.sin(psi + theta) + Alpha

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

# INFERENCE
for idx in range(len(GT_poses)-1):

    action_id = 1
    if idx >= 4:
        action_id = 3
    if idx >= 21:
        action_id = 1
    if idx >= 24:
        action_id = 2

    # Execute action
    action_generated = action_generator(actions[action_id], action_noise_diag, no_noise=False)
    lambda_belief.action_step(action_generated, action_noise)

    # Defining opening angle
    opening_angle = 50
    opening_angle_rad = opening_angle * math.pi / 180

    geo_measurements = list()
    sem_measurements = list()
    for lambda_real in range(len(belief_list)):
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

        if np.abs(xy_angle - psi) <= opening_angle_rad and geo_radius <= measurement_radius_limit:

            GT_DA.append(j + 1)

            geo_cov = np.diag(geo_model_noise_diag)
            meas_relative_pose = geo_measurement_generator(geo_part, geo_cov, no_noise=False)

            geo_measurements.append(meas_relative_pose)
            # print(geo_measurements)

            for lambda_real in range(len(belief_list)):

                if class_GT[j + 1] == 2:
                    sem_part = sem_measurements_generator_2(GT_objects[j].between(GT_poses[idx + 1]),
                                                            no_noise=False)
                    #print('Object: ' + str(j + 1) + ',sem: ' + str(sem_part) + ',robot ' + robot_id + ',Class 2')
                else:
                    sem_part = sem_measurements_generator(GT_objects[j].between(GT_poses[idx + 1]),
                                                          no_noise=False)

                sem_measurements[lambda_real].append(sem_part)  # GENERATED
                # print(sem_part)

    lambda_belief.add_measurements(geo_measurements, sem_measurements,
                                                da_current_step=GT_DA,
                                                number_of_samples=num_samp,
                                                new_input_object_prior=None,
                                                new_input_object_covariance=None)
    print('Number of realizations: ' + str(len(lambda_belief.belief_list[0].belief_matrix)))
    #if idx > 5:
    print('Dirichlet parameters: ' + str(lambda_belief.object_lambda_prob_dirichlet(3)))

name = 'plan_10'

ax = lambda_belief.belief_list[0].graph_realizations_in_one(idx, fig_num=0, show_obj=True,
                                                          show_weights=False, show_plot=False)
plotterdaac2d.GT_plotter_line(GT_poses, GT_objects, fig_num=0, ax=ax, show_plot=True,
                              plt_save_name=None, pause_time=None, limitx=[-10, 7], limity=[-8, 8],
                              red_point=idx, jpg_save=False)