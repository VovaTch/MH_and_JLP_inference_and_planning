from __future__ import print_function
import numpy as np
import itertools
import gtsam
import math
import copy
import plotterdaac2d
import colorsys
import matplotlib.pyplot as plt
from gaussianb_isam2 import GaussianBelief
import lambdab
#from gaussianb import GaussianBelief
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class CombinedReward:

    def __init__(self, goal_pose = gtsam.Pose2(gtsam.Pose3(0.0, 0.0, 0.0)), goal_weight=0,
                 information_theoretic_weight=0, lambda_uncertainty_weight=0):

        self.goal_pose = goal_pose
        self.goal_weight = goal_weight
        self.information_theoretic_weight = information_theoretic_weight
        self.lambda_uncertainty_weight = lambda_uncertainty_weight

    # Compute the reward; passed to the planner
    def compute_reward(self, lambda_belief):
        """

        :type lambda_belief: lambdab.LambdaBelief
        """
        reward = 0

        # Uncertainty in Lambda in the belief
        if self.lambda_uncertainty_weight != 0:
            reward += -lambda_belief.entropy_realizations() * self.lambda_uncertainty_weight

        # Distance to goal, need to decide the metric
        if self.goal_weight != 0:
            pass

        # Information theoretic, need to decide the metric
        if self.information_theoretic_weight != 0:
            pass

        return reward


