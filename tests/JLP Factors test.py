from __future__ import print_function
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, O, L
import lambda_prior_factor
#import joint_lambda_pose_factor_2d as joint_lambda_pose_factor_fake_2d
import joint_lambda_pose_factor_2d as joint_lambda_pose_factor_fake_2d

#import libJointLambdaPoseFactor
#import libJointLambdaPoseFactorFake
#import gaussianb_jlp

# libLambdaPriorFactor
# Symbol initialization
#X = lambda i: X(i)
XO = lambda j: O(j)
Lambda = lambda k: L(k)
# joint_lambda_pose_factor_fake_2d.initialize_python_objects()
joint_lambda_pose_factor_fake_2d.initialize_python_objects('../exp_model_Book Jacket.pt',
                                                           '../exp_model_Digital Clock.pt',
                                                           '../exp_model_Packet.pt',
                                                           '../exp_model_Pop Bottle.pt',
                                                           '../exp_model_Soap Dispenser.pt',
                                                           '../rinf_model_Book Jacket.pt',
                                                           '../rinf_model_Digital Clock.pt',
                                                           '../rinf_model_Packet.pt',
                                                           '../rinf_model_Pop Bottle.pt',
                                                           '../rinf_model_Soap Dispenser.pt')

# Graph initialization
graph_1 = gtsam.NonlinearFactorGraph()

# Add prior lambda
prior_noise = gtsam.noiseModel.Diagonal.Covariance(np.array([[3.2, 0.0, 0.0, 0.0],
                                                            [0.0, 3.2, 0.0, 0.0],
                                                            [0.0, 0.0, 3.2, 0.0],
                                                            [0.0, 0.0, 0.0, 3.2]]))
prior_noise_pose = gtsam.noiseModel.Diagonal.Variances(np.array([0.003, 0.003, 0.001, 0.003, 0.003, 0.003]))
geo_noise = gtsam.noiseModel.Diagonal.Variances(np.array([0.003, 0.003, 0.001, 0.002, 0.002, 0.003]))
graph_1.add(lambda_prior_factor.LambdaPriorFactor(L(0), np.array([0., 0., 0., 0.]), prior_noise))
graph_1.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)), prior_noise_pose))
graph_1.add(gtsam.BetweenFactorPose3(X(0), XO(1), gtsam.Pose3(gtsam.Pose2(1.0, 0.0, -0 * 3.14/2)), geo_noise))
graph_1.add(joint_lambda_pose_factor_fake_2d.JLPFactor(X(0), XO(1), Lambda(0), Lambda(1), [1.0, 0.0, 0.0, 0.0],
                                                       [1.1, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 1.1, 0.0, 1.1]))
print(graph_1)

initial = gtsam.Values()
initial.insert(Lambda(0), [0.0, 0.0, 0.0, 0.0])
initial.insert(Lambda(1), [0.0, 0.0, 0.0, 0.0])
initial.insert(X(0), gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0), np.array([0.036, 1.72645, 0.0])))
initial.insert(XO(1), gtsam.Pose3(gtsam.Rot3.Ypr(0 *3.140 / 4, 0.0, 0.0), np.array([0.0, 0.0, 0.0])))

params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph_1, initial, params)
result = optimizer.optimize()
print(result.atVector(Lambda(0)))
print(result.atVector(Lambda(1)))
marginals = gtsam.Marginals(graph_1, result)
L0_marginal_movariance = marginals.marginalCovariance(Lambda(0))
L1_marginal_movariance = marginals.marginalCovariance(Lambda(1))

print(L0_marginal_movariance)
print(L1_marginal_movariance)
#print()

#print(joint_lambda_pose_factor_fake_2d.AuxOutputsJLP.predictions(gtsam.Pose3(gtsam.Pose2(0.7, -3.1, 0.21))))



