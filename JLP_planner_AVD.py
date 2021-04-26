from __future__ import print_function
import numpy as np
import gtsam
import geomodel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class JLPPLannerPrimitives:

    def __init__(self, JLP_Belief, AVD_mapping, AVD_poses, entropy_lower_limit=None):

        # Initiate the belief
        self.lambda_belief = JLP_Belief

        # AVD initializations
        self.AVD_mapping = AVD_mapping
        self.AVD_poses = AVD_poses

        # Initiate entropy lower limit
        self.entropy_lower_limit = entropy_lower_limit

    # Planning session for know primitives, pre-defined constant set after 1st wave
    def planning_session(self, action_noise, geo_noise, current_location_name, horizon=1, ML_planning=True,
                         number_of_samples=10,
                         belief_for_planning=None, motion_primitives=None, reward_print=False, enable_MCTS=False,
                         MCTS_braches_per_action=2, return_index_flag=False, custom_reward_function=None,
                         sample_use_flag=False, reward_mode=1, use_lower_bound = False,  number_of_beliefs=None):

        possible_actions = ['forward','rotate_ccw','rotate_cw','backward','left','right']
        function_motion_primitives = list()
        function_motion_primitives_idx = list()
        for next_action_name in possible_actions:
            if self.AVD_mapping[current_location_name][next_action_name]:
                function_motion_primitives.append(self.AVD_poses[current_location_name].
                                                  between(self.AVD_poses[self.AVD_mapping[current_location_name][next_action_name]]))
                function_motion_primitives_idx.append(next_action_name)

        #################################################
        # Single planning sample per belief
        if ML_planning is True:
            # Initialize reward
            reward = -np.inf
            best_action_idx = -1

            # Run over all primitives
            for prim_idx in range(len(function_motion_primitives)):

                if belief_for_planning is None:

                    belief_for_planning = self.lambda_belief

                # Create a propagated belief
                candidate_lambda_belief = belief_for_planning.clone()
                prev_belief = belief_for_planning.clone()

                # Propagate the belief and generate measurements
                geo_ML, sem_ML = candidate_lambda_belief.\
                    generate_future_measurements(function_motion_primitives[prim_idx], action_noise,
                                                 ML_geo=ML_planning, ML_cls=ML_planning)

                geo_meas_list = list(geo_ML.values())
                sem_exp_list = list(sem_ML[0].values())
                sem_cov_list = list(sem_ML[1].values())
                da_step = list(sem_ML[0].keys())

                # Use samples instead of parameters directly
                if sample_use_flag is True:
                    for idx_obj in range(len(sem_exp_list)):
                        sem_sample = np.random.multivariate_normal(sem_exp_list[idx_obj],
                                                                   sem_cov_list[idx_obj], number_of_samples)
                        sem_exp_sampled = np.sum(sem_sample) / number_of_samples
                        distance_from_expectation = sem_sample - sem_exp_sampled
                        sem_cov_sampled = np.matmul(np.transpose(distance_from_expectation),
                                                    distance_from_expectation) \
                                          / (number_of_samples - 1)
                        sem_exp_list[idx_obj] = [sem_exp_sampled]
                        sem_cov_list[idx_obj] = np.matrix(sem_cov_sampled)

                candidate_lambda_belief.action_step(function_motion_primitives[prim_idx], action_noise)
                candidate_lambda_belief.add_measurements(geo_meas_list, sem_exp_list, sem_cov_list, da_step)

                candidate_reward = 0
                if reward_mode == 3:
                    for obj in candidate_lambda_belief.object_lambda_dict:
                        if use_lower_bound is True:
                            candidate_reward_temp, _ = prev_belief. \
                                lambda_entropy_individual_bounds(prev_belief.find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit)
                            candidate_reward += candidate_reward_temp
                        else:
                            candidate_reward += prev_belief. \
                                        lambda_entropy_individual_numeric(prev_belief.find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit)
                # For horizon larger than 1, call the function recursively
                if horizon > 1 and enable_MCTS is False:
                    try:
                        _, candidate_reward_temp = self.planning_session(action_noise, geo_noise, horizon=horizon-1,
                                                                    ML_planning=ML_planning,
                                                                    motion_primitives=function_motion_primitives,
                                                                    number_of_samples=number_of_samples,
                                                                    reward_print=False, enable_MCTS=enable_MCTS,
                                                                    belief_for_planning=candidate_lambda_belief,
                                                                    sample_use_flag=sample_use_flag,
                                                                    reward_mode=reward_mode,
                                                                    use_lower_bound=use_lower_bound,
                                                                    current_location_name=self.AVD_mapping
                                                                    [current_location_name]
                                                                    [function_motion_primitives_idx[prim_idx]])
                        candidate_reward += candidate_reward_temp

                        for obj in candidate_lambda_belief.object_lambda_dict:
                            if reward_mode == 1 or reward_mode == 3:
                                candidate_reward -= candidate_lambda_belief. \
                                    lambda_entropy_individual_numeric(candidate_lambda_belief.find_latest_key(obj),
                                                                      entropy_lower_limit=self.entropy_lower_limit)
                            elif reward_mode == 2:
                                candidate_reward -= candidate_lambda_belief. \
                                    lambda_expectation_entropy(candidate_lambda_belief.find_latest_key(obj))

                    except:
                        candidate_reward = - np.inf

                elif horizon > 1 and enable_MCTS is True:

                    # Select n follow up actions per each parent action, and build a path from those.
                    pick_idx = np.random.choice(len(function_motion_primitives), MCTS_braches_per_action, replace=False)
                    pick_idx = np.sort(pick_idx)
                    function_motion_primitives_reduced = [function_motion_primitives[i] for i in list(pick_idx)]

                    try:
                        _, candidate_reward_temp = self.planning_session(action_noise, geo_noise, horizon=horizon - 1,
                                                                    ML_planning=ML_planning,
                                                                    motion_primitives=function_motion_primitives_reduced,
                                                                    number_of_samples=number_of_samples,
                                                                    reward_print=False, enable_MCTS=enable_MCTS,
                                                                    MCTS_braches_per_action=1,
                                                                    belief_for_planning=candidate_lambda_belief,
                                                                    sample_use_flag=sample_use_flag,
                                                                    reward_mode=reward_mode,
                                                                    use_lower_bound=use_lower_bound,
                                                                    current_location_name=self.AVD_mapping
                                                                    [current_location_name]
                                                                    [function_motion_primitives_idx[prim_idx]])
                        candidate_reward += candidate_reward_temp

                        for obj in candidate_lambda_belief.object_lambda_dict:
                            if reward_mode == 1 or reward_mode == 3:
                                if use_lower_bound is True:
                                    candidate_reward_temp, _ = candidate_lambda_belief. \
                                        lambda_entropy_individual_bounds(candidate_lambda_belief.find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit)
                                    candidate_reward -= candidate_reward_temp
                                else:
                                    candidate_reward -= candidate_lambda_belief. \
                                        lambda_entropy_individual_numeric(candidate_lambda_belief.find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit)
                            elif reward_mode == 2:
                                candidate_reward -= candidate_lambda_belief. \
                                    lambda_expectation_entropy(candidate_lambda_belief.find_latest_key(obj))
                    except:
                        candidate_reward = -np.inf

                elif horizon == 1:

                    for obj in candidate_lambda_belief.object_lambda_dict:
                        if reward_mode == 1 or reward_mode == 3:
                            if use_lower_bound is True:
                                candidate_reward_temp, _ = candidate_lambda_belief. \
                                                        lambda_entropy_individual_bounds(
                                    candidate_lambda_belief.find_latest_key(obj),
                                    entropy_lower_limit=self.entropy_lower_limit)
                                candidate_reward -= candidate_reward_temp
                            else:
                                candidate_reward -= candidate_lambda_belief. \
                                                        lambda_entropy_individual_numeric(
                                    candidate_lambda_belief.find_latest_key(obj),
                                    entropy_lower_limit=self.entropy_lower_limit)
                        elif reward_mode == 2:
                            candidate_reward -= candidate_lambda_belief. \
                                                    lambda_expectation_entropy(candidate_lambda_belief.
                                                                               find_latest_key(obj))

                if reward_print is True:
                    print('Horizon left: ' + str(horizon) + '; Motion primitive: ' +
                          function_motion_primitives_idx[prim_idx] +
                          '; Candidate Reward: ' + str(candidate_reward))

                if candidate_reward > reward:
                    reward = candidate_reward
                    best_action_idx = prim_idx

                del candidate_lambda_belief

        #####################################
        elif ML_planning is False:

            # Initialize reward
            reward = -np.inf
            best_action_idx = -1

            for prim_idx in range(len(function_motion_primitives)):

                candidate_reward = 0
                # Create a propagated belief
                candidate_lambda_belief = [None] * number_of_samples
                prev_belief = [None] * number_of_samples
                reward_collector = list()

                if belief_for_planning is None:
                    for idx in range(number_of_samples):
                        candidate_lambda_belief[idx] = self.lambda_belief.clone()
                        prev_belief[idx] = self.lambda_belief.clone()
                else:
                    for idx in range(number_of_samples):
                        candidate_lambda_belief[idx] = belief_for_planning.clone()
                        prev_belief[idx] = belief_for_planning.clone()

                for idx in range(number_of_samples):

                    geo_ML, sem_ML = candidate_lambda_belief[idx].\
                        generate_future_measurements(function_motion_primitives[prim_idx], action_noise,
                                                     ML_geo=ML_planning, ML_cls=ML_planning, geo_noise=geo_noise)
                    # TODO: COULD CAUSE ERRORS

                    geo_meas_list = list(geo_ML.values())
                    sem_exp_list = list(sem_ML[0].values())
                    sem_cov_list = list(sem_ML[1].values())
                    da_step = list(sem_ML[0].keys())

                    if sample_use_flag is True:
                        for idx_obj in range(len(sem_exp_list)):
                            sem_sample = np.random.multivariate_normal(sem_exp_list[idx_obj],
                                                                       sem_cov_list[idx_obj], number_of_samples)
                            sem_exp_sampled = np.sum(sem_sample) / number_of_samples
                            distance_from_expectation = sem_sample - sem_exp_sampled
                            sem_cov_sampled = np.matmul(np.transpose(distance_from_expectation),
                                                        distance_from_expectation) \
                                              / (number_of_samples - 1)
                            sem_exp_list[idx_obj] = [sem_exp_sampled]
                            sem_cov_list[idx_obj] = np.matrix(sem_cov_sampled)

                    candidate_lambda_belief[idx].action_step(function_motion_primitives[prim_idx], action_noise)
                    candidate_lambda_belief[idx].add_measurements(geo_meas_list, sem_exp_list, sem_cov_list, da_step)

                # If horizon limit is reached, i.e. horizon is 1, compute the entropy reward.
                # If not, compute the reward of the next step recursively.
                candidate_reward = 0
                if reward_mode == 3:
                    for obj in candidate_lambda_belief[0].object_lambda_dict:
                        if use_lower_bound is True:
                            candidate_reward_temp, _= prev_belief[0]. \
                                        lambda_entropy_individual_bounds(prev_belief[0].find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit)
                            candidate_reward += candidate_reward_temp
                        else:
                            candidate_reward += prev_belief[0]. \
                                lambda_entropy_individual_numeric(prev_belief[0].find_latest_key(obj),
                                                                  entropy_lower_limit=self.entropy_lower_limit)

                if horizon > 1 and enable_MCTS is False:
                    try:
                        for idx in range(number_of_samples):
                            _, candidate_reward_sub = self.planning_session(action_noise, geo_noise, horizon=horizon - 1,
                                                                            ML_planning=ML_planning,
                                                                            motion_primitives=function_motion_primitives,
                                                                            number_of_samples=number_of_samples,
                                                                            reward_print=False, enable_MCTS=enable_MCTS,
                                                                            belief_for_planning=candidate_lambda_belief[
                                                                                idx],
                                                                            custom_reward_function=custom_reward_function,
                                                                            sample_use_flag=sample_use_flag,
                                                                            reward_mode=reward_mode,
                                                                            use_lower_bound=use_lower_bound,
                                                                    current_location_name=self.AVD_mapping
                                                                    [current_location_name]
                                                                    [function_motion_primitives_idx[prim_idx]])

                            candidate_reward += candidate_reward_sub / number_of_samples

                            for obj in candidate_lambda_belief[idx].object_lambda_dict:
                                if reward_mode == 1 or reward_mode == 3:
                                    if use_lower_bound is True:
                                        candidate_reward_temp, _ = candidate_lambda_belief[idx]. \
                                                                lambda_entropy_individual_bounds(
                                            int(candidate_lambda_belief[idx].find_latest_key(obj)),
                                                                          entropy_lower_limit=self.entropy_lower_limit)
                                        candidate_reward -= candidate_reward_temp / number_of_samples
                                        reward_collector.append(candidate_reward_temp)
                                    else:
                                        candidate_reward -= candidate_lambda_belief[idx]. \
                                                                lambda_entropy_individual_numeric(
                                            int(candidate_lambda_belief[idx].find_latest_key(obj)),
                                                                          entropy_lower_limit=self.entropy_lower_limit) \
                                                            / number_of_samples
                                        reward_collector.append(
                                            - candidate_lambda_belief[idx].lambda_entropy_individual_numeric(
                                                int(candidate_lambda_belief[idx].find_latest_key(obj)),
                                                                          entropy_lower_limit=self.entropy_lower_limit))
                                elif reward_mode == 2:
                                    candidate_reward -= candidate_lambda_belief[idx].lambda_expectation_entropy(
                                        candidate_lambda_belief[idx].find_latest_key(obj)) / number_of_samples
                                    reward_collector.append(- candidate_lambda_belief[idx].lambda_expectation_entropy(
                                        candidate_lambda_belief[idx].find_latest_key(obj)))

                    except:
                        candidate_reward = -np.inf
                        reward_collector.append(-np.inf)

                elif horizon > 1 and enable_MCTS is True:  # TODO: IMPLEMENT A TRUE MCTS

                    # Select n follow up actions per each parent action, and build a path from those.
                    pick_idx = np.random.choice(len(function_motion_primitives_idx), MCTS_braches_per_action, replace=False)
                    pick_idx = np.sort(pick_idx)
                    function_motion_primitives_reduced = [function_motion_primitives_idx[i] for i in list(pick_idx)]

                    try:
                        for idx in range(number_of_samples):
                            _, candidate_reward_sub = self.planning_session(action_noise, geo_noise, horizon=horizon - 1,
                                                                        ML_planning=ML_planning,
                                                                        motion_primitives=function_motion_primitives_reduced,
                                                                        number_of_samples=number_of_samples,
                                                                        reward_print=False, enable_MCTS=enable_MCTS,
                                                                        MCTS_braches_per_action=1,
                                                                        belief_for_planning=candidate_lambda_belief[idx],
                                                                        custom_reward_function=custom_reward_function,
                                                                        reward_mode=reward_mode,
                                                                        use_lower_bound=use_lower_bound,
                                                                    current_location_name=self.AVD_mapping
                                                                    [current_location_name]
                                                                    [function_motion_primitives_idx[prim_idx]])

                            candidate_reward += candidate_reward_sub / number_of_samples

                            for obj in candidate_lambda_belief[idx].object_lambda_dict:
                                if reward_mode == 1 or reward_mode == 3:
                                    if use_lower_bound is True:
                                        candidate_reward_temp,_ = candidate_lambda_belief[idx].lambda_entropy_individual_bounds(
                                            candidate_lambda_belief[idx].find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit)
                                        candidate_reward -= candidate_reward_temp / number_of_samples
                                        reward_collector.append(candidate_reward_temp)
                                    else:
                                        candidate_reward -= candidate_lambda_belief[idx].lambda_entropy_individual_numeric(
                                            candidate_lambda_belief[idx].find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit) \
                                                            / number_of_samples
                                        reward_collector.append(- candidate_lambda_belief[idx].
                                                                lambda_entropy_individual_numeric(
                                            candidate_lambda_belief[idx].find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit))
                                elif reward_mode == 2:
                                    candidate_reward -= candidate_lambda_belief[idx]. \
                                                            lambda_expectation_entropy(
                                        candidate_lambda_belief[idx].find_latest_key(obj)) / number_of_samples
                                    reward_collector.append(
                                        - candidate_lambda_belief[idx].lambda_expectation_entropy(
                                            candidate_lambda_belief[idx].find_latest_key(obj)))
                    except:
                        candidate_reward = -np.inf
                        reward_collector.append(-np.inf)

                elif horizon == 1:

                    #candidate_reward_sub = 0
                    for idx in range(number_of_samples):

                        for obj in candidate_lambda_belief[idx].object_lambda_dict:
                            if reward_mode == 1 or reward_mode == 3:
                                if use_lower_bound is True:
                                    candidate_reward_temp, _ = candidate_lambda_belief[idx]. \
                                                            lambda_entropy_individual_bounds(
                                        candidate_lambda_belief[idx].find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit)
                                    candidate_reward -= candidate_reward_temp / number_of_samples
                                    reward_collector.append(candidate_reward_temp)
                                else:
                                    candidate_reward -= candidate_lambda_belief[idx]. \
                                                            lambda_entropy_individual_numeric(
                                        candidate_lambda_belief[idx].find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit) \
                                                        / number_of_samples
                                    reward_collector.append(
                                        - candidate_lambda_belief[idx].lambda_entropy_individual_numeric(
                                            candidate_lambda_belief[idx].find_latest_key(obj),
                                                                          entropy_lower_limit=self.entropy_lower_limit))
                            elif reward_mode == 2:
                                candidate_reward -= candidate_lambda_belief[idx]. \
                                                        lambda_expectation_entropy(candidate_lambda_belief[idx].
                                                                                   find_latest_key(obj)) / \
                                                    number_of_samples
                                reward_collector.append(
                                    - candidate_lambda_belief[idx].lambda_expectation_entropy(
                                        candidate_lambda_belief[idx].find_latest_key(obj)))

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
            return function_motion_primitives_idx[best_action_idx], reward


        # Enter a pre-made trajectory to compute costs
    def evaluate_trajectory(self, action_list, action_noise, geo_noise, ML_planning=True,
                            number_of_samples=10, belief_for_planning=None, return_sub_costs=False,
                            lambda_plots=False, lambda_plots_object=1, CVaR_flag=False, sample_use_flag=False,
                            num_samples=50, reward_mode=1, number_of_beliefs=None):

        if ML_planning is True:

            if belief_for_planning is None:
                belief_for_planning = self.lambda_belief

            # Create a propagated belief
            candidate_lambda_belief = belief_for_planning.clone()
            prev_belief = belief_for_planning.clone()

            # Propagate the belief and generate measurements
            geo_ML, sem_ML = candidate_lambda_belief.generate_future_measurements(action_list[0], action_noise,
                                                                                  ML_geo=ML_planning,
                                                                                  ML_cls=ML_planning)

            geo_meas_list = list(geo_ML.values())
            sem_exp_list = list(sem_ML[0].values())
            sem_cov_list = list(sem_ML[1].values())
            da_step = list(sem_ML[0].keys())

            candidate_lambda_belief.action_step(action_list[0], action_noise)
            candidate_lambda_belief.add_measurements(geo_meas_list, sem_exp_list, sem_cov_list, da_step)

            candidate_reward = 0
            if reward_mode == 3:
                for obj in belief_for_planning.object_lambda_dict:
                    candidate_reward += prev_belief. \
                        lambda_entropy_individual_numeric(prev_belief.find_latest_key(obj),
                                                          entropy_lower_limit=self.entropy_lower_limit)

            if len(action_list) > 1:
                try:
                    candidate_reward = self.evaluate_trajectory(action_list[1:], action_noise,
                                                                geo_noise, ML_planning=ML_planning,
                                                                belief_for_planning=candidate_lambda_belief,
                                                                CVaR_flag=CVaR_flag, reward_mode=reward_mode)
                    for obj in candidate_lambda_belief.daRealization[-1]:
                        if reward_mode == 1 or reward_mode == 3:
                            candidate_reward -= candidate_lambda_belief.\
                                lambda_entropy_individual_numeric(candidate_lambda_belief.find_latest_key(obj),
                                                                      entropy_lower_limit=self.entropy_lower_limit)
                        elif reward_mode == 2:
                            candidate_reward -= candidate_lambda_belief.\
                                lambda_expectation_entropy(candidate_lambda_belief.find_latest_key(obj))
                except:
                    candidate_reward = -np.inf

            else:
                for obj in candidate_lambda_belief.object_lambda_dict:
                    if reward_mode == 1 or reward_mode == 3:
                        candidate_reward -= candidate_lambda_belief. \
                            lambda_entropy_individual_numeric(candidate_lambda_belief.find_latest_key(obj),
                                                                      entropy_lower_limit=self.entropy_lower_limit)
                    elif reward_mode == 2:
                        candidate_reward -= candidate_lambda_belief. \
                            lambda_expectation_entropy(candidate_lambda_belief.find_latest_key(obj))

            if lambda_plots is True:
                candidate_lambda_belief.simplex_2class(lambda_plots_object, show_plot=False)
                plt.show()

            return candidate_reward

        elif ML_planning is False:

            candidate_reward = 0
            if reward_mode == 3:
                for obj in belief_for_planning.object_lambda_dict:
                    candidate_reward += belief_for_planning. \
                        lambda_entropy_individual_numeric(belief_for_planning.find_latest_key(obj),
                                                          entropy_lower_limit=self.entropy_lower_limit)
            # Create a propagated belief
            candidate_lambda_belief = [None] * number_of_samples
            prev_belief = [None] * number_of_samples
            reward_collector = list()

            if belief_for_planning is None:
                for idx in range(number_of_samples):
                    candidate_lambda_belief[idx] = self.lambda_belief.clone()
                    candidate_lambda_belief[idx] = self.lambda_belief.clone()
            else:
                for idx in range(number_of_samples):
                    prev_belief[idx] = belief_for_planning.clone()
                    prev_belief[idx] = belief_for_planning.clone()

            for idx in range(number_of_samples):

                geo_ML, sem_ML = candidate_lambda_belief[idx].generate_future_measurements(action_list[0], action_noise,
                                                                                      ML_geo=ML_planning,
                                                                                      ML_cls=ML_planning,
                                                                                          geo_noise=geo_noise) #TODO: COULD CAUSE ERRORS

                geo_meas_list = list(geo_ML.values())
                sem_exp_list = list(sem_ML[0].values())
                sem_cov_list = list(sem_ML[1].values())
                da_step = list(sem_ML[0].keys())

                if sample_use_flag is True:
                    for idx_obj in range(len(sem_exp_list)):

                        sem_sample = np.random.multivariate_normal(sem_exp_list[idx_obj],
                                                                   sem_cov_list[idx_obj], number_of_samples)
                        sem_exp_sampled = np.sum(sem_sample) / number_of_samples
                        distance_from_expectation = sem_sample - sem_exp_sampled
                        sem_cov_sampled = np.matmul(np.transpose(distance_from_expectation), distance_from_expectation)\
                                          / (number_of_samples - 1)
                        sem_exp_list[idx_obj] = [sem_exp_sampled]
                        sem_cov_list[idx_obj] = np.matrix(sem_cov_sampled)

                candidate_lambda_belief[idx].action_step(action_list[0], action_noise)
                candidate_lambda_belief[idx].add_measurements(geo_meas_list, sem_exp_list, sem_cov_list, da_step)

                if len(action_list) > 1:
                    try:
                        candidate_reward_temp = self.evaluate_trajectory(action_list[1:], action_noise,
                                                                    geo_noise, ML_planning=ML_planning,
                                                                    belief_for_planning=candidate_lambda_belief[idx],
                                                                    CVaR_flag=CVaR_flag,
                                                                    number_of_samples=number_of_samples,
                                                                    reward_mode=reward_mode)
                        candidate_reward += candidate_reward_temp
                        for obj in candidate_lambda_belief[idx].object_lambda_dict:
                            if reward_mode == 1 or reward_mode == 3:
                                candidate_reward -= candidate_lambda_belief[idx]. \
                                    lambda_entropy_individual_numeric(candidate_lambda_belief[idx].find_latest_key(obj))\
                                                    / number_of_samples
                                reward_collector.append(- candidate_lambda_belief[idx].lambda_entropy_individual_numeric(
                                    candidate_lambda_belief[idx].find_latest_key(obj)))
                            elif reward_mode == 2:
                                candidate_reward -= candidate_lambda_belief[idx]. \
                                                        lambda_expectation_entropy(
                                    candidate_lambda_belief[idx].find_latest_key(obj)) \
                                                    / number_of_samples
                                reward_collector.append(
                                    - candidate_lambda_belief[idx].lambda_expectation_entropy(
                                        candidate_lambda_belief[idx].find_latest_key(obj)))
                    except:
                        candidate_reward = -np.inf
                        reward_collector.append(-np.inf)

                else:
                    for obj in candidate_lambda_belief[idx].object_lambda_dict:
                        if reward_mode == 1 or reward_mode == 3:
                            candidate_reward -= candidate_lambda_belief[idx]. \
                                                    lambda_entropy_individual_numeric(
                                candidate_lambda_belief[idx].find_latest_key(obj))
                            reward_collector.append(- candidate_lambda_belief[idx].lambda_entropy_individual_numeric(
                                candidate_lambda_belief[idx].find_latest_key(obj)))
                        elif reward_mode == 2:
                            candidate_reward -= candidate_lambda_belief[idx]. \
                                                    lambda_expectation_entropy(
                                candidate_lambda_belief[idx].find_latest_key(obj))
                            reward_collector.append(
                                - candidate_lambda_belief[idx].lambda_expectation_entropy(
                                    candidate_lambda_belief[idx].find_latest_key(obj)))

            # Lambda plots flag
            if lambda_plots is True:
                for can_idx in range(len(candidate_lambda_belief) - 1):
                    candidate_lambda_belief[can_idx].simplex_2class(lambda_plots_object, show_plot=False,
                                                                    alpha_g=1.5 / (
                                                                            len(candidate_lambda_belief) - 1))
                plt.show()

            # CVaR cost computation
            CVaR_cost = 0
            sub_costs_np = np.array(reward_collector)
            if CVaR_flag is True:
                CVaR_threshold = 0.05
                sub_costs_np = np.sort(sub_costs_np)
                last_index = int(CVaR_threshold * len(candidate_lambda_belief[-1].daRealization))
                sub_costs_cut = sub_costs_np[0:last_index + 1]
                CVaR_cost = np.average(sub_costs_cut)

            if return_sub_costs is False:
                if CVaR_flag is True:
                    return CVaR_cost
                else:
                    return candidate_reward
            else:
                return reward_collector

    # Rotate pose covariance matrix
    @staticmethod
    def rotate_cov_6x6(cov_matrix, rotation_3x3):

        rotation_6x6 = np.eye(6)
        rotation_6x6[3:6,3:6] = rotation_3x3
        rot_1 = np.matmul(rotation_6x6, cov_matrix)
        rotated_cov_matrix = np.matmul(rot_1, np.transpose(rotation_6x6))

        return rotated_cov_matrix