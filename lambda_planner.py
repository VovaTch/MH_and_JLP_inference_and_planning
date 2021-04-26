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

class LambdaPlannerPrimitives:

    def __init__(self, lambda_belief, motion_primitives=None, entropy_lower_limit=None):

        # Initialize with a lambda belief
        self.lambda_belief = lambda_belief

        # Initialize motion primitives if not given
        if motion_primitives is None:

            self.motion_primitives = list()
            self.motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0.00, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
            self.motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0.08, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
            self.motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(-0.04, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
            self.motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.04, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))
            self.motion_primitives.append(gtsam.Pose3(gtsam.Rot3.Ypr(0.08, 0.0, 0.0), np.array([0.25, 0.0, 0.0])))

        # Get motion primitives if given
        else:

            self.motion_primitives = motion_primitives

        self.entropy_lower_limit = entropy_lower_limit

    # Planning session for know primitives, pre-defined constant set after 1st wave TODO: CHECK THIS THING
    def planning_session(self, action_noise, geo_noise, horizon=1, ML_planning=True, motion_primitives=None,
                         number_of_samples = 10, reward_print=False, enable_MCTS=False, MCTS_braches_per_action=2,
                         belief_for_planning=None, return_index_flag=False, custom_reward_function=None, reward_mode=1,
                         sample_use_flag=False, use_lower_bound=False):

        if motion_primitives is None:
            function_motion_primitives = self.motion_primitives
        else:
            function_motion_primitives = motion_primitives

        if ML_planning is True: #TODO: ML planning is false
            # Initialize reward
            reward = -np.inf
            best_action_idx = -1

            for prim_idx in range(len(function_motion_primitives)):

                if belief_for_planning is None:

                    belief_for_planning = self.lambda_belief

                candidate_lambda_belief = belief_for_planning.clone()
                candidate_lambda_belief.belief_list = [x.clone() for x in belief_for_planning.belief_list]

                # Cloning necesity, otherwise the lambda belief will be overwritten
                for belief_idx in range(len(belief_for_planning.belief_list)):
                    candidate_lambda_belief.belief_list[belief_idx].belief_matrix = dict()
                    for g_idx in belief_for_planning.belief_list[belief_idx].belief_matrix:
                        candidate_lambda_belief.belief_list[belief_idx].belief_matrix[g_idx] = \
                            belief_for_planning.belief_list[belief_idx].belief_matrix[g_idx].clone()

                for belief in candidate_lambda_belief.belief_list:
                    object_list = belief.obj_realization
                    break

                # Find average geometric measurement (assume for now that cov matrices are very similar to each other)
                # TODO: Change to something that doesnt assume that
                geo_meas = dict()
                rotation_avg = dict()
                for ind_belief in range(len(candidate_lambda_belief.belief_list)):
                    geo_meas_ind, _ = candidate_lambda_belief.belief_list[ind_belief].generate_future_measurements(
                        function_motion_primitives[prim_idx], action_noise, geo_noise)

                    for key in geo_meas_ind:

                        # TODO: THIS KIND OF AVERAGING DOES NOT WORK. IF NECESSARY, USE QUATERNION SLERP AND IMPLEMENT IT
                        rotation_avg[key] = geo_meas_ind[key].rotation() #Average rotating, doing a workaround
                        geo_meas_ind_avgpart = gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.0),
                                                           np.array([geo_meas_ind[key].x() / len(candidate_lambda_belief.
                                                                                               belief_list),
                                                                        geo_meas_ind[key].y() / len(candidate_lambda_belief.
                                                                                               belief_list),
                                                                        geo_meas_ind[key].z() / len(candidate_lambda_belief.
                                                                                               belief_list)]))
                        if key not in geo_meas:
                            geo_meas[key] = geo_meas_ind_avgpart
                        else:
                            geo_meas[key] = geo_meas[key].compose(geo_meas_ind_avgpart)

                for key in geo_meas_ind:
                    geo_meas[key] = geo_meas[key].compose(gtsam.Pose3(rotation_avg[key], np.array([0.0, 0.0, 0.0])))

                # Go over all individual hybrid beliefs in lambda belief
                for ind_belief in range(len(candidate_lambda_belief.belief_list)):
                    _, sem_meas = candidate_lambda_belief.belief_list[ind_belief].generate_future_measurements(
                        function_motion_primitives[prim_idx],action_noise, geo_noise, ML_cls=False,
                        overwrite_geo=geo_meas)
                    geo_meas_list = list(geo_meas.values())
                    sem_meas_list = list(sem_meas.values())
                    da_step = list(sem_meas.keys())

                    # Make sure the propagation doesn't fail
                    while True:
                        try:

                            candidate_lambda_belief.belief_list[ind_belief].action_step(function_motion_primitives[prim_idx], action_noise)
                            candidate_lambda_belief.belief_list[ind_belief].add_measurements(geo_meas_list, sem_meas_list, da_step,
                                                        number_of_samples=number_of_samples)
                            break

                        except:

                            candidate_reward = -np.inf # Dunno if it will work
                            break

                # If horizon limit is reached, i.e. horizon is 1, compute the entropy reward.
                # If not, compute the reward of the next step recursively.
                candidate_reward = 0
                if reward_mode == 3:
                    for obj in object_list:
                        candidate_reward -= belief_for_planning.entropy(obj,
                                                                        entropy_lower_limit=self.entropy_lower_limit)

                if horizon > 1 and enable_MCTS is False:
                    try:
                        _, candidate_reward = self.planning_session(action_noise, geo_noise, horizon=horizon-1,
                                                                    ML_planning=ML_planning,
                                                                    motion_primitives=function_motion_primitives,
                                                                    number_of_samples=number_of_samples,
                                                                    reward_print=False, enable_MCTS=enable_MCTS,
                                                                    belief_for_planning=candidate_lambda_belief,
                                                                    reward_mode=reward_mode)
                    except:
                        candidate_reward = -np.inf
                elif horizon > 1 and enable_MCTS is True: #TODO: IMPLEMENT A TRUE MCTS

                    # Select n follow up actions per each parent action, and build a path from those.
                    pick_idx = np.random.choice(len(self.motion_primitives), MCTS_braches_per_action, replace=False)
                    pick_idx = np.sort(pick_idx)
                    function_motion_primitives_reduced = [self.motion_primitives[i] for i in list(pick_idx)]

                    try:
                        _, candidate_reward = self.planning_session(action_noise, geo_noise, horizon=horizon - 1,
                                                                    ML_planning=ML_planning,
                                                                    motion_primitives=function_motion_primitives_reduced,
                                                                    number_of_samples=number_of_samples,
                                                                    reward_print=False, enable_MCTS=enable_MCTS,
                                                                    MCTS_braches_per_action=1,
                                                                    belief_for_planning=candidate_lambda_belief,
                                                                    reward_mode=reward_mode)
                    except:
                        candidate_reward = -np.inf

                if reward_mode == 1 or reward_mode == 3:
                    for obj in object_list:
                        candidate_reward += -candidate_lambda_belief.\
                            entropy(obj, entropy_lower_limit=self.entropy_lower_limit)
                elif reward_mode == 2:
                    for obj in object_list:
                        candidate_reward += -candidate_lambda_belief.entropy_lambda_expectation(obj)
                # candidate_reward += -candidate_lambda_belief.avg_entropy()

                if reward_print is True:
                    print('Horizon left: ' + str(horizon) + '; Motion primitive: ' + str(prim_idx + 1) +
                          '; Candidate Reward: ' + str(candidate_reward))

                if candidate_reward > reward:
                    reward = candidate_reward
                    best_action_idx = prim_idx

                del candidate_lambda_belief



        # MORE MATHEMATICALLY ACCURATE ----------------------------------------------------------

        if ML_planning is False:
            # Initialize reward
            reward = -np.inf
            best_action_idx = -1

            for prim_idx in range(len(function_motion_primitives)):

                if belief_for_planning is None:
                    belief_for_planning = self.lambda_belief

                # Create a series of candidate beliefs, each propagated with sampled gammas
                candidate_lambda_belief = [None] * (belief_for_planning.number_of_beliefs + 1)
                for can_idx in range(belief_for_planning.number_of_beliefs + 1):

                    candidate_lambda_belief[can_idx] = belief_for_planning.clone()
                    candidate_lambda_belief[can_idx].belief_list = [x.clone() for x in belief_for_planning.belief_list]

                    # Cloning necesity, otherwise the lambda belief will be overwritten
                    for belief_idx in range(len(belief_for_planning.belief_list)):
                        candidate_lambda_belief[can_idx].belief_list[belief_idx].belief_matrix = dict()
                        for g_idx in belief_for_planning.belief_list[belief_idx].belief_matrix:
                            candidate_lambda_belief[can_idx].belief_list[belief_idx].belief_matrix[g_idx] = \
                                belief_for_planning.belief_list[belief_idx].belief_matrix[g_idx].clone()

                for belief in candidate_lambda_belief[0].belief_list:
                    object_list = belief.obj_realization
                    break

                # Go over all individual hybrid beliefs in lambda belief
                for ind_belief in range(len(candidate_lambda_belief[-1].belief_list)):
                    for gamma_sample_idx in range(belief_for_planning.number_of_beliefs):
                        geo_meas, sem_meas = candidate_lambda_belief[-1].belief_list[ind_belief].generate_future_measurements(
                            function_motion_primitives[prim_idx], action_noise, geo_noise, ML_cls=False)
                        geo_meas_list = list(geo_meas.values())
                        sem_meas_list = list(sem_meas.values())
                        da_step = list(sem_meas.keys())

                        candidate_lambda_belief[ind_belief].belief_list[gamma_sample_idx].action_step(function_motion_primitives[prim_idx],
                                                                                    action_noise)
                        candidate_lambda_belief[ind_belief].belief_list[gamma_sample_idx].add_measurements(geo_meas_list, sem_meas_list,
                                                                                         da_step,
                                                                                         number_of_samples=number_of_samples)

                # If horizon limit is reached, i.e. horizon is 1, compute the entropy reward.
                # If not, compute the reward of the next step recursively.
                candidate_reward = 0
                if reward_mode == 3:
                    for obj in object_list:
                        candidate_reward -= belief_for_planning.entropy(obj,
                                                                        entropy_lower_limit=self.entropy_lower_limit)

                if horizon > 1 and enable_MCTS is False:
                    try:
                        for cat_idx in range(belief_for_planning.number_of_beliefs):
                            _, candidate_reward_sub = self.planning_session(action_noise, geo_noise, horizon=horizon - 1,
                                                                        ML_planning=ML_planning,
                                                                        motion_primitives=function_motion_primitives,
                                                                        number_of_samples=number_of_samples,
                                                                        reward_print=False, enable_MCTS=enable_MCTS,
                                                                        belief_for_planning=candidate_lambda_belief[cat_idx],
                                                                        custom_reward_function=custom_reward_function,
                                                                        reward_mode=reward_mode)
                            candidate_reward += candidate_reward_sub / belief_for_planning.number_of_beliefs

                    except:
                        candidate_reward = -np.inf
                elif horizon > 1 and enable_MCTS is True:  # TODO: IMPLEMENT A TRUE MCTS

                    # Select n follow up actions per each parent action, and build a path from those.
                    pick_idx = np.random.choice(len(self.motion_primitives), MCTS_braches_per_action, replace=False)
                    pick_idx = np.sort(pick_idx)
                    function_motion_primitives_reduced = [self.motion_primitives[i] for i in list(pick_idx)]

                    try:
                        for cat_idx in range(belief_for_planning.number_of_beliefs):
                            _, candidate_reward_sub = self.planning_session(action_noise, geo_noise, horizon=horizon - 1,
                                                                        ML_planning=ML_planning,
                                                                        motion_primitives=function_motion_primitives_reduced,
                                                                        number_of_samples=number_of_samples,
                                                                        reward_print=False, enable_MCTS=enable_MCTS,
                                                                        MCTS_braches_per_action=1,
                                                                        belief_for_planning=candidate_lambda_belief[cat_idx],
                                                                        custom_reward_function=custom_reward_function,
                                                                        reward_mode=reward_mode)
                            candidate_reward += candidate_reward_sub / belief_for_planning.number_of_beliefs
                    except:
                        candidate_reward = -np.inf

                if reward_mode == 1 or reward_mode == 3:
                    for can_idx in range(len(candidate_lambda_belief) - 1):
                        for obj in object_list:
                            candidate_reward += -candidate_lambda_belief[can_idx].\
                                entropy(obj, entropy_lower_limit=self.entropy_lower_limit) / belief_for_planning.\
                                number_of_beliefs
                elif reward_mode == 2:
                    for can_idx in range(len(candidate_lambda_belief) - 1):
                        for obj in object_list:
                            candidate_reward += -candidate_lambda_belief.entropy_lambda_expectation(obj) / \
                                                belief_for_planning.number_of_beliefs
                # candidate_reward += -candidate_lambda_belief.avg_entropy()

                if reward_print is True:
                    print('Horizon left: ' + str(horizon) + '; Motion primitive: ' + str(prim_idx + 1) +
                          '; Candidate Reward: ' + str(candidate_reward))

                if candidate_reward > reward:
                    reward = candidate_reward
                    best_action_idx = prim_idx

                del candidate_lambda_belief

        if return_index_flag is False:
            return function_motion_primitives[best_action_idx], reward
        else:
            return best_action_idx, reward

    # Enter a pre-made trajectory to compute costs
    def evaluate_trajectory(self, action_list, action_noise, geo_noise, ML_planning=True,
                            number_of_samples=10, belief_for_planning=None, return_sub_costs=False,
                            lambda_plots=False, lambda_plots_object=1, CVaR_flag=False, reward_mode=1):


        if ML_planning is False:
            # Initialize reward
            reward = -np.inf
            best_action_idx = -1

            if belief_for_planning is None:
                belief_for_planning = self.lambda_belief

            # Create a series of candidate beliefs, each propagated with sampled gammas
            candidate_lambda_belief = [None] * (belief_for_planning.number_of_beliefs + 1)
            for can_idx in range(belief_for_planning.number_of_beliefs + 1):

                candidate_lambda_belief[can_idx] = belief_for_planning.clone()
                candidate_lambda_belief[can_idx].belief_list = [x.clone() for x in belief_for_planning.belief_list]

                # Cloning necesity, otherwise the lambda belief will be overwritten
                for belief_idx in range(len(belief_for_planning.belief_list)):
                    candidate_lambda_belief[can_idx].belief_list[belief_idx].belief_matrix = dict()
                    for g_idx in belief_for_planning.belief_list[belief_idx].belief_matrix:
                        candidate_lambda_belief[can_idx].belief_list[belief_idx].belief_matrix[g_idx] = \
                            belief_for_planning.belief_list[belief_idx].belief_matrix[g_idx].clone()

            # Go over all individual hybrid beliefs in lambda belief
            for ind_belief in range(len(candidate_lambda_belief[-1].belief_list)):
                for gamma_sample_idx in range(belief_for_planning.number_of_beliefs):
                    geo_meas, sem_meas = candidate_lambda_belief[-1].belief_list[
                        ind_belief].generate_future_measurements(
                        action_list[0], action_noise, geo_noise, ML_cls=False, ML_geo=False)
                    geo_meas_list = list(geo_meas.values())
                    sem_meas_list = list(sem_meas.values())
                    da_step = list(sem_meas.keys())

                    candidate_lambda_belief[ind_belief].belief_list[gamma_sample_idx].action_step(
                        action_list[0],
                        action_noise)
                    candidate_lambda_belief[ind_belief].belief_list[gamma_sample_idx].add_measurements(
                        geo_meas_list, sem_meas_list,
                        da_step,
                        number_of_samples=number_of_samples)

            # If horizon limit is reached, i.e. horizon is 1, compute the entropy reward.
            # If not, compute the reward of the next step recursively.
            candidate_reward = 0
            if reward_mode == 3:
                candidate_reward = belief_for_planning.entropy_realizations()

            sub_costs = list()

            if len(action_list) > 1:
                try:
                    for cat_idx in range(belief_for_planning.number_of_beliefs):
                        candidate_reward_sub = self.evaluate_trajectory(action_list[1:], action_noise,
                                                                        geo_noise, ML_planning=ML_planning,
                                                                        belief_for_planning=candidate_lambda_belief[cat_idx],
                                                                        CVaR_flag=CVaR_flag, reward_mode=reward_mode)
                        candidate_reward += candidate_reward_sub / belief_for_planning.number_of_beliefs
                        if reward_mode == 1 or reward_mode == 3:
                            candidate_reward -= candidate_lambda_belief[-1].entropy_realizations() \
                                                / belief_for_planning.number_of_beliefs
                        elif reward_mode == 2:
                            candidate_reward -= candidate_lambda_belief[-1].entropy_lambda_expectation_total() \
                                                / belief_for_planning.number_of_beliefs

                        sub_costs.append(candidate_reward_sub)

                except:
                    candidate_reward = -np.inf

            else:
                for can_idx in range(len(candidate_lambda_belief) - 1):
                    if reward_mode == 1 or reward_mode == 3:
                        candidate_reward += -candidate_lambda_belief[can_idx].entropy_realizations() \
                                            / belief_for_planning.number_of_beliefs
                    sub_costs.append(-candidate_lambda_belief[can_idx].entropy_realizations())

            # Lambda plots flag
            if lambda_plots is True:
                for can_idx in range(len(candidate_lambda_belief) - 1):
                    candidate_lambda_belief[can_idx].simplex_2class(lambda_plots_object, show_plot=False,
                                                                    alpha_g=1.5/(len(candidate_lambda_belief) - 1))
                plt.show()

            CVaR_cost = 0
            sub_costs_np = np.array(sub_costs)
            if CVaR_flag is True:
                CVaR_threshold = 0.05
                sub_costs_np = np.sort(sub_costs_np)
                last_index = int(CVaR_threshold * belief_for_planning.number_of_beliefs)
                sub_costs_cut = sub_costs_np[0:last_index + 1]
                CVaR_cost = np.average(sub_costs_cut)

            if return_sub_costs is False:
                if CVaR_flag is True:
                    return CVaR_cost
                else:
                    return candidate_reward
            else:
                return sub_costs







