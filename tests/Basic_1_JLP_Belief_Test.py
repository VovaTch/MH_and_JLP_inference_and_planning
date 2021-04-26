from __future__ import print_function
import numpy as np
import math
import gtsam
import geomodel
import damodel
import clsUmodel_fake_1d as clsUmodel
import gaussianb_jlp
import matplotlib as plt

# Initializations
np.random.seed(10)
opening_angle = 50
opening_angle_rad = opening_angle * math.pi / 180
measurement_radius_limit = 10
measurement_radius_minimum = 0

# Models
np.set_printoptions(precision=3)
plt.rcParams.update({'font.size': 16})

da_model = damodel.DAModel(camera_fov_angle_horizontal=opening_angle_rad, range_limit=measurement_radius_limit,
                           range_minimum=measurement_radius_minimum)
geo_model_noise_diag = np.array([0.00001, 0.00001, 0.001, 0.01, 0.01, 0.00001])
geo_model_noise = gtsam.noiseModel.Diagonal.Variances(geo_model_noise_diag)
geo_model = geomodel.GeoModel(geo_model_noise)
lambda_model = clsUmodel.JLPModel()

# Robot priors
prior = gtsam.Pose3(gtsam.Pose2(-2.0, 0.0, 0.0))
prior_noise_diag = np.array([00.000001, 0.000001, 0.0000038, 00.00002, 00.0000202, 0.000001])
prior_noise = gtsam.noiseModel.Diagonal.Variances(prior_noise_diag)

# Action list
actions = list()
action_noise_diag = np.array([0.0003, 0.0003, 0.01, 0.03, 0.030, 0.001])
action_noise = gtsam.noiseModel.Diagonal.Variances(action_noise_diag)

actions.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
actions.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
actions.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.04, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
actions.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0.04, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
actions.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.08, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
actions.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0.08, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))

# Object GT
GT_objects = list()
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi/2, 0.0, 0.0), np.array([1.0, 0.5, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi/2, 0.0, 0.0), np.array([1.0, -0.5, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi/4, 0.0, 0.0), np.array([0.0, 0.5, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi/4, 0.0, 0.0), np.array([-.0, -0.5, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(math.pi/2, 0.0, 0.0), np.array([1.0, 1.5, 0.0])))
GT_objects.append(gtsam.Pose3(gtsam.Rot3.Ypr(-math.pi/2, 0.0, 0.0), np.array([1.0, -1.5, 0.0])))


# Initialize JLP belief ------------------------------------------------
JLP_belief = gaussianb_jlp.JLPBelief(geo_model, da_model, lambda_model, prior, prior_noise, cls_enable=True,
                                     lambda_prior_mean=np.array([0.]), lambda_prior_noise=((0.5)))


# Measurement and action generation
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

GT_pose = prior

# INFERENCE
for idx in range(10):

    # Execute action
    action_generated = action_generator(actions[0], action_noise_diag, no_noise=False)
    JLP_belief.action_step(action_generated, action_noise)
    GT_pose = GT_pose.compose(actions[0])

    # Defining opening angle
    opening_angle = 50
    opening_angle_rad = opening_angle * math.pi / 180

    geo_measurements = list()
    sem_measurements_exp = list()
    sem_measurements_cov = list()
    GT_DA = list()

    # Generate measurements and forward the beliefs
    for j in range(len(GT_objects)):

        # Check if the object is seen
        xy_angle = np.arctan2(GT_objects[j].y() - GT_pose.y(),
                              GT_objects[j].x() - GT_pose.x())
        angles = GT_pose.rotation().matrix()
        psi = np.arctan2(angles[1, 0], angles[0, 0])
        geo_part = GT_pose.between(GT_objects[j])
        geo_radius = np.sqrt(geo_part.x() ** 2 + geo_part.y() ** 2 + geo_part.z() ** 2)

        if np.abs(xy_angle - psi) <= opening_angle_rad and geo_radius <= measurement_radius_limit:

            GT_DA.append(j + 1)

            geo_cov = np.diag(geo_model_noise_diag)
            meas_relative_pose = geo_measurement_generator(geo_part, geo_cov, no_noise=False)

            geo_measurements.append(meas_relative_pose)
            # print(geo_measurements)
            sem_measurements_exp.append([0.3])
            sem_measurements_cov.append(np.array([[0.5]]))

    JLP_belief.add_measurements(geo_measurements, sem_measurements_exp, sem_measurements_cov, GT_DA)


JLP_belief.print_results(show_entropy=True)
#JLP_belief.simplex_3class(1, show_plot=True, log_likelihood=False)
JLP_belief.display_graph()





