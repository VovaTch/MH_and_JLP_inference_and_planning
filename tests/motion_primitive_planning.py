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
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from JLP_planner import JLPPLannerPrimitives
from hybridb import HybridBelief
from lambdab_lg import LambdaBelief
from lambda_planner import LambdaPlannerPrimitives
import time

# Mode choosing
Lambda_BSP_mode = 1  # 1: Multi-Hybrid. 2: JLP.
reward_mode = 2 #1: Entropy #2: Expectation entropy #3: Information Gain
inf_planning_mode = 2 #1: Inference #2: Planning
use_bounds = False

# Initializations ----------------------------------------------------
conf_flag = False
sample_use_flag = True
ML_planning_flag = True
track_length = 20
horizon = 12
ML_update = True
MCTS_flag = True
MCTS_branches = 4
cls_enable = True
display_graphs = True
measurement_radius_limit = 10
measurement_radius_minimum = 0
com_radius = 100000
class_GT = {1: 2, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2}#, 10: 1}
GT_cls_realization = ((1, 1), (2, 2), (3, 1), (4, 1), (5, 1), (6, 2), (7, 1), (8, 1))
np.random.seed(282)
measurements_enabler = True
opening_angle = 60
opening_angle_rad = opening_angle * math.pi / 180
number_of_beliefs = 1
num_samp = 50
CVaR_flag = True
start_time = time.time()
entropy_lower_limit = -5
random_object_location_flag = False

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
if Lambda_BSP_mode == 1:
    if conf_flag is True:
        import clsmodel_lg1_conf as clsmodel_lg1
    else:
        import clsmodel_lg1 as clsmodel_lg1
    cls_model = clsmodel_lg1.ClsModel()
if Lambda_BSP_mode == 2:
    if conf_flag is True:
        import clsUmodel_fake_1d_conf as clsUmodel_fake_1d
    else:
        import clsUmodel_fake_1d as clsUmodel_fake_1d
    cls_model = clsUmodel_fake_1d.JLPModel()
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

if random_object_location_flag is False:
    if inf_planning_mode == 1:
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([2.5, -4.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(3.141/2, 0.0, 0.0), np.array([4, -7.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141/2, 0.0, 0.0), np.array([4.5, -1.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141/4, 0.0, 0.0), np.array([1.0, -4.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0*3.141, 0.0, 0.0), np.array([7.5, 4.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141/4, 0.0, 0.0), np.array([2, -2, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141/2, 0.0, 0.0), np.array([8.5, 4.7, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141/4, 0.0, 0.0), np.array([11.5, 1.0, 0.0])))
        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-2*3.141/4, 0.0, 0.0), np.array([12, -6.5, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-2*3.141/4, 0.0, 0.0), np.array([14.5, 2.7, 0.0])))
    elif inf_planning_mode == 2:

        GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(3 * 3.141 / 2, 0.0, 0.0), np.array([10.5, 0.0, 0.0])))

        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([2.5, -4.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(3.141/2, 0.0, 0.0), np.array([4, -7.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141/2, 0.0, 0.0), np.array([4.5, -1.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(3.141/4, 0.0, 0.0), np.array([6.5, 3.0, 0.0])))
        # # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0*3.141, 0.0, 0.0), np.array([7.5, 4.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(1*3.141/2, 0.0, 0.0), np.array([8, 1.5, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141/2, 0.0, 0.0), np.array([8.5, 7.7, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-3.141/4, 0.0, 0.0), np.array([11.5, 1.0, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(0*3.141/4, 0.0, 0.0), np.array([12, -2.5, 0.0])))
        # GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-2*3.141/4, 0.0, 0.0), np.array([14.5, 2.7, 0.0])))


else:
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

# Actions for inference
action_inf = list()
action_inf.append(gtsam.Pose3(gtsam.Pose2(1.0, 0.0, -np.pi / 16)))
action_inf.append(gtsam.Pose3(gtsam.Pose2(1.0, 0.0, np.pi / 16)))


# Robot poses
GT_poses = list()
GT_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0*math.pi/2, 0.0, 0.0), np.array([-4.0, 0.0, 0.0])))

if inf_planning_mode == 1:
    for idx in range(5):
        GT_poses.append(GT_poses[-1].compose(gtsam.Pose3(gtsam.Pose2(1.0, 0.0, -np.pi / 16))))
    for idx in range(10):
        GT_poses.append(GT_poses[-1].compose(gtsam.Pose3(gtsam.Pose2(1.0, 0.0, np.pi / 16))))
    for idx in range(5):
        GT_poses.append(GT_poses[-1].compose(gtsam.Pose3(gtsam.Pose2(1.0, 0.0, -np.pi / 16))))

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
                         cls_enable=cls_enable, pruning_threshold=15, ML_update=ML_update))

    lambda_belief = LambdaBelief(belief_list)
    lambda_planner = LambdaPlannerPrimitives(lambda_belief, entropy_lower_limit=entropy_lower_limit)

if Lambda_BSP_mode == 2:

    lambda_belief = gaussianb_jlp.JLPBelief(geo_model, da_model, cls_model, GT_poses[0], prior_noise,
                                                          lambda_prior_mean=lambda_prior,
                                                          lambda_prior_noise=lambda_prior_noise_diag,
                                                          cls_enable=cls_enable)
    lambda_planner = JLPPLannerPrimitives(lambda_belief, entropy_lower_limit=entropy_lower_limit)

# Plot all the GT sequences
fig = plt.figure(0)
ax = fig.gca()
plotterdaac2d.GT_plotter_line(GT_poses, GT_objects, color=(0.8, 0.4, 0.4),
                              limitx=[-10, 8], limity=[-15, 5], show_plot=False, ax=ax,
                              start_pose=GT_poses[0], robot_id='Robot')
ax.set_xlabel('X axis [m]')
ax.set_ylabel('Y axis [m]')
plt.tight_layout()
wedges = []
obj_idx_wedge = 0
for GT_object in GT_objects:
    obj_idx_wedge += 1
    if class_GT[obj_idx_wedge] == 1 or conf_flag is False:
        wedge = mpatches.Wedge((GT_object.x(), GT_object.y()), measurement_radius_limit,
                               GT_object.rotation().yaw() * 180 / 3.141 - 15,
                               GT_object.rotation().yaw() * 180 / 3.141 + 15, ec="none")
    else:
        wedge = mpatches.Wedge((GT_object.x(), GT_object.y()), measurement_radius_limit,
                               GT_object.rotation().yaw() * 180 / 3.141 - 15 + 180,
                               GT_object.rotation().yaw() * 180 / 3.141 + 15 + 180, ec="none")
    wedges.append(wedge)
collection = PatchCollection(wedges, facecolors='blue', alpha=0.2)
ax.add_collection(collection)
ax.set_xlim(-8,15)
ax.set_ylim(-10,10)
plt.show()

# Robot poses reset
GT_poses = list()
GT_poses.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0*math.pi/2, 0.0, 0.0), np.array([-4.0, 0.0, 0.0])))

# SEMANTIC MEASUREMENT GENERATION FUNCTION-----------------------------------------------------

def sem_measurements_generator(relative_pose, no_noise=False):

    radius = np.sqrt(relative_pose.x()**2 + relative_pose.y()**2)
    psi = np.arctan2(relative_pose.y(), relative_pose.x())
    theta = np.arctan2(-relative_pose.z(), radius)

    K = 2
    if conf_flag is True:
        Alpha = 0.3
    else:
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
    if conf_flag is True:
        Alpha = -0.3
    else:
        Alpha = -0.5

    if conf_flag is True:
        R = K * (0.7 - 0.3 * np.cos(psi + theta))
    else:
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
MSDE = list()
MSDE.append(1/4)

current_entropy_list_per_object = dict()
current_MSDE_list_per_object = dict()
for obj in class_GT:
    current_entropy_list_per_object[obj] = list()
    current_entropy_list_per_object[obj].append(0)
    current_MSDE_list_per_object[obj] = list()
    current_MSDE_list_per_object[obj].append(1/4)
MSDE_per_object = list()
sigma_list = list()

Simplex_flag = False

#for sequence_idx in range(track_length):
for idx in range(track_length):

    if inf_planning_mode == 2:
        action, reward = lambda_planner.planning_session(action_noise, np.diag(geo_model_noise_diag), horizon=horizon, reward_print=True,
                                                         enable_MCTS=MCTS_flag, MCTS_braches_per_action=MCTS_branches,
                                                         ML_planning=ML_planning_flag, sample_use_flag=sample_use_flag,
                                                         reward_mode=reward_mode, motion_primitives=motion_primitives,
                                                         number_of_samples=number_of_beliefs, use_lower_bound=use_bounds)
    elif inf_planning_mode == 1:

        if idx < 5:
            action = action_inf[0]
        elif idx < 15:
            action = action_inf[1]
        else:
            action = action_inf[0]

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

    for obj in class_GT:
        current_entropy_list_per_object[obj].append(0)
        current_MSDE_list_per_object[obj].append(1/4)

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

    if Lambda_BSP_mode is 1:
        mh_jlp = 'MH'
    elif Lambda_BSP_mode is 2 and number_of_beliefs > 1:
        mh_jlp = 'JLP'
    elif Lambda_BSP_mode is 2 and number_of_beliefs == 1:
        mh_jlp = 'WEU'

    if GT_DA or Simplex_flag is True:

        Simplex_flag = True
        lambda_belief.simplex_2class(1, show_plot=False)
        plt.tight_layout()
        plt.savefig('../figures/' + mh_jlp + '_' + f'{idx + 1:03}' + '_Simplex.png')
        plt.clf()

        lambda_belief.lambda_bar_error_graph(show_plt=False)
        plt.tight_layout()
        plt.savefig('../figures/' + mh_jlp + '_' + f'{idx + 1:03}' + '_Bars.png')
        plt.clf()



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

plotterdaac2d.GT_plotter_line(GT_poses, GT_objects, fig_num=0, ax=ax, show_plot=True,
                              plt_save_name=None, pause_time=None, limitx=[-2, 15], limity=[-8, 8],
                              red_point=idx + 1, jpg_save=False)

fig = plt.figure(111)
print(current_entropy_list)
plt.plot(range(len(GT_poses)), current_entropy_list)
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
    plt.plot(range(len(GT_poses)), current_entropy_list_per_object[obj], alpha=2 / np.sqrt(len(class_GT)))
    plt.text(len(GT_poses)-2, current_entropy_list_per_object[obj][-1], 'O' + str(obj))
#plt.title('Entropy over time per object')
plt.xlabel('Time step')
plt.ylabel('Entropy')
plt.xlim(0, track_length)
plt.grid(True)
plt.tight_layout()
plt.show()

fig_2 = plt.figure(333)
plt.plot(range(len(GT_poses)), MSDE)
#plt.title('Classification accuracy MSDE over time')
plt.xlabel('Time step')
plt.ylabel('MSDE')
#plt.yscale('log')
plt.ylim(0, 1)
plt.xlim(0, track_length)
plt.grid(True)
plt.tight_layout()
if Lambda_BSP_mode is 1 and number_of_beliefs > 1:
    plt.savefig('../figures/' + passive_active + '_MSDE_MH' + str(number_of_beliefs) + '.png')
elif Lambda_BSP_mode is 1 and number_of_beliefs == 1:
    plt.savefig('../figures/' + passive_active + '_MSDE_WEU.png')
elif Lambda_BSP_mode is 2:
    plt.savefig('../figures/' + passive_active + '_MSDE_JLP.png')
plt.show()

fig = plt.figure(334)
print(current_entropy_list)
for obj in class_GT:
    plt.plot(range(len(GT_poses)), current_MSDE_list_per_object[obj], alpha=2 / np.sqrt(len(class_GT)))
    plt.text(len(GT_poses) - 2, current_MSDE_list_per_object[obj][-1], 'O' + str(obj))
#plt.title('MSDE over time per object')
plt.xlabel('Time step')
plt.ylabel('MSDE')
#plt.yscale('log')
plt.ylim(0, 1)
plt.xlim(0, track_length)
plt.grid(True)
#plt.tight_layout()
if Lambda_BSP_mode is 1 and number_of_beliefs > 1:
    plt.savefig('../figures/' + passive_active + '_MSDE_MH' + str(number_of_beliefs) + '_ind.png')
elif Lambda_BSP_mode is 1 and number_of_beliefs == 1:
    plt.savefig('../figures/' + passive_active + '_MSDE_WEU_ind.png')
elif Lambda_BSP_mode is 2:
    plt.savefig('../figures/' + passive_active + '_MSDE_JLP_ind.png')
plt.show()

fig_2 = plt.figure(444)
plt.plot(range(len(GT_poses)-1), sigma_list)
plt.title('Sigma width over time')
plt.xlabel('Time step')
plt.ylabel('Sigma')
plt.grid(True)
plt.tight_layout()
plt.show()