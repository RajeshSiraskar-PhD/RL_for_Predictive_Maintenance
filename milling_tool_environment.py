# Milling Tool Environment
# - Handles multiple dataset. PHM and NUAA
# - Default Rewards: ```R1: +1.0, R2: -4.0, R3: -0.5```

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

NO_ACTION = 0
REPLACE = 1

class Milling_Tool_Env(gym.Env):
    
        def __init__(self, max_records=0, env_type = '', rul_threshold=0.0, R1=1.0, R2=0.0, R3=0.0):
            print(f'-- Ver. RB 0.5. Tested 26-Nov-24: Environment: {env_type} | RUL threshold: {rul_threshold:4.3f} | Records: {max_records} | R1: {R1:3.1f}, R2: {R2:3.1f}, R3: {R3:3.1f}')

            # Initialize
            self.env_type = env_type
            self.R1 = R1
            self.R2 = R2
            self.R3 = R3
            self.reward = 0.0
            self.cummulative_rewards = 0.0

            self.a_rewards = []
            self.a_actions = []
            self.a_action_recommended = []
            self.a_rul = []
            self.a_time_since_last_replacement = []

            self.df = None
            self.current_time_step = 0
            self.max_records = max_records
            self.maintenance_cost = 0.0
            self.replacement_events = 0
            self.time_since_last_replacement = 0
            self.rul_threshold = rul_threshold

            high = np.array(6*[1.0], dtype=np.float32)
            low = np.array(6*[-1.0], dtype=np.float32)
            self.observation_space = spaces.Box(low, high, dtype=np.float32)
            self.action_space = spaces.Discrete(2)

        ## Add tool wear data-set
        ## Keep rul_threshold intialized to self.rul_threshold
        def tool_wear_data(self, df, rul_threshold=0.0):
            self.df = df
            self.rul_threshold = rul_threshold
            self.max_records = len(df.index)
            print(f'   * Tool-wear data updated: {self.max_records}. RUL threshold: {self.rul_threshold:4.3f}')

        def _get_observation(self):
            if (self.df is not None):
                if (self.env_type == 'PHM'):
                    obs_values = np.array([
                        self.df.loc[self.current_time_step, 'force_x'],
                        self.df.loc[self.current_time_step, 'force_y'],
                        self.df.loc[self.current_time_step, 'force_z'],
                        self.df.loc[self.current_time_step, 'vibration_x'],
                        self.df.loc[self.current_time_step, 'vibration_y'],
                        self.df.loc[self.current_time_step, 'vibration_z']
                    ], dtype=np.float32)
                else:
                    obs_values = np.array([
                        self.df.loc[self.current_time_step, 'axial_force'],
                        self.df.loc[self.current_time_step, 'force_z'],
                        self.df.loc[self.current_time_step, 'vibration_x'],
                        self.df.loc[self.current_time_step, 'vibration_y'],
                        self.df.loc[self.current_time_step, 'vibration1'],
                        self.df.loc[self.current_time_step, 'vibration2']
                    ], dtype=np.float32)
            else:
                obs_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            observation = obs_values.flatten()
            return observation

        def _get_auxilliary_info(self):
            if (self.df is not None):
                recommended_action = int(self.df.loc[self.current_time_step, 'ACTION_CODE'])
                rul = float(self.df.loc[self.current_time_step, 'RUL'])
            else:
                recommended_action = 0
                rul = 0.0

            return recommended_action, rul

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            # self.a_rewards = []
            # self.a_actions = []
            # self.a_action_recommended = []
            # self.a_rul = []

            # Choose the tool wear at a random time (spatial) location from a uniformly random distribution
            # self.current_time_step = np.random.randint(0, int(RANDOM_TOOL_START_OF_LIFE * self.max_records), 1, dtype=int)
            self.current_time_step = 0
            self.reward = 0.0
            self.cummulative_rewards = 0.0

            observation = self._get_observation()
            info = {'reset':'Reset'}
            return observation, info

        ## Step
        def step(self, action):
            terminated = False
            reward = 0.0
            info = {'Step':'-'}
            # Get auxilliary info: current RUL reading (note this is NOT part of the observation) and the expert's recommended action
            recommended_action, self.rul = self._get_auxilliary_info()
            self.maintenance_cost = 0.0

            # Check if Episode over
            if self.current_time_step >= self.max_records:
                done = True

            # Reward for RUL maximized and penalize if breached
            if self.rul > (1.0+EARLY_DETECT_FACTOR)*self.rul_threshold:
                reward = self.current_time_step*self.R1
            else:
                reward = self.current_time_step*self.R2

            # If tool replaced, add cost of replacement
            if action == REPLACE:
                # Update time_since_last_replacement
                self.time_since_last_replacement = self.current_time_step
                self.a_time_since_last_replacement.append(self.current_time_step)
                reward = self.current_time_step*self.R3
                self.current_time_step = 0

            self.reward = reward
            self.cummulative_rewards = self.cummulative_rewards + reward

            # Information arrays
            self.a_rewards.append(reward)
            self.a_actions.append(action)
            self.a_action_recommended.append(recommended_action)
            self.a_rul.append(self.rul)

            if (action != REPLACE) and (self.current_time_step < (self.max_records-1)):
              self.current_time_step += 1

            # Action taken, reward set for that action, now take in next observation
            observation = self._get_observation()


            if self.rul <= self.rul_threshold:
              self.cummulative_rewards = 0.0

            # if self.rul > self.rul_threshold:
            #     print(f'Action>> Expert:{recommended_action:2d} Agent:{action:2d}| RUL: {self.rul:>5.2f} | Reward: {reward:>7.2f} -- CR: {self.cummulative_rewards:>7.2f}' )
            # else:
            #     print(f'*** RUL reached: {self.rul:3.2f} ***')
            #     self.cummulative_rewards = 0.0

            # writer.add_scalar('reward', reward)
            return observation, self.reward, terminated, False, info