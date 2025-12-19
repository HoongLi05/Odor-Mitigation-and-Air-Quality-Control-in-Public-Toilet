import streamlit as st
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, SAC, A2C
from stable_baselines3.common.monitor import Monitor
import itertools
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import plotly.graph_objects as go
import plotly.subplots as sp


# -------------------------
# Configuration
# -------------------------
LOAD_MODEL = False

system_parameters = {
    # Power ratings in Watts
    'POWER_EXHAUST_FAN_W': 55,
    'POWER_CEILING_FAN_W': 40,
    'POWER_DEHUMIDIFIER_W': 500,

    # Threshold values for environmental parameters
    'THRESHOLD_CO2_PPM_MIN': 800,
    'THRESHOLD_CO2_PPM_MID': 1000,
    'THRESHOLD_CO2_PPM_MAX': 1500,

    # NH3 Thresholds
    'NH3_THRESHOLD_PPM_MIN': 0.032,
    'NH3_THRESHOLD_PPM_MAX': 1.6,

    # H2S Thresholds
    'H2S_THRESHOLD_PPM_MIN': 0.01,
    'H2S_THRESHOLD_PPM_MAX': 1.5,

    'THRESHOLD_TEMPERATURE_C': {
        'LOW': 28,                # Degrees Celsius
        'HIGH': 30.5              # Degrees Celsius
    },
    'THRESHOLD_HUMIDITY_PERCENT': {
        'LOW': 61.4,              # Percentage
        'HIGH': 73.8              # Percentage
    },


    # Scaling coefficients (k values) for reward functions
    'K_CO2': 0.05,
    'K_ENERGY_CONSUMPTION': 0.02,
    'K_TEMPERATURE_COMFORT': 0.01,
    'K_HUMIDITY_COMFORT': 0.01,
    'K_NH3': 0.05,
    'K_H2S': 0.05,
    'DELTA_T_HOURS': 1/60,       # Timestep duration in hours for energy calculation
}

print("System parameters and constants defined successfully.")
print(system_parameters)


# -------------------------
# K-Value Tuning Functions
# -------------------------
def normalize_k_values():
  """K value normalized based on benchmark value"""
  benchmarks = {
      'CO2': 500.0,
      'NH3': 1.4,
      'H2S': 0.5,
      'Temperature': 3.0,
      'Humidity': 15.0,
      'Energy': 150
  }

  target_range = 1.0
  k_values = {}

  #Normalization of secondary penalties
  k_values['K_CO2'] = target_range / (benchmarks['CO2'] ** 1.5)
  k_values['K_NH3'] = target_range / (benchmarks['NH3'] ** 1.5)
  k_values['K_H2S'] = target_range / (benchmarks['H2S'] ** 1.5)
  # Standardize key names for consistency
  k_values['K_TEMPERATURE_COMFORT'] = target_range / (benchmarks['Temperature'] ** 1.5)
  k_values['K_HUMIDITY_COMFORT'] = target_range / (benchmarks['Humidity'] ** 1.5)
  k_values['K_ENERGY_CONSUMPTION'] = target_range / (benchmarks['Energy'] ** 1.5)

  return k_values

def set_k_by_priority():
  """K value is set based on priority
  Priority 1-5, 5 being the highest"""

  priorities = {
      'CO2': 3,
      'NH3': 5,
      'H2S': 5,
      'TEMPERATURE_COMFORT': 2,
      'HUMIDITY_COMFORT': 2,
      'ENERGY_CONSUMPTION':1
  }

  priority_to_k = {
      1: 0.01,
      2: 0.02,
      3: 0.05,
      4: 0.1,
      5: 0.2
  }

  k_params = {}
  for param, priority in priorities.items():
    k_params[f'K_{param}'] = priority_to_k[priority]

  return k_params

def update_k_values(method='priority'):
  """Update the K value in the system parameters"""
  if method == 'normalize':
    new_k_values = normalize_k_values()
  elif method == 'priority':
    new_k_values = set_k_by_priority()
  elif method == 'hybrid':
    # Hybrid approach: Prioritize for security-related aspects, normalize for others.
    priority_k = set_k_by_priority()
    normalize_k = normalize_k_values()
    new_k_values = {
        'K_NH3': priority_k['K_NH3'], # Use string literal for key
        'K_H2S': priority_k['K_H2S'], # Use string literal for key
        'K_CO2': priority_k['K_CO2'], # Use string literal for key
        'K_TEMPERATURE_COMFORT': normalize_k['K_TEMPERATURE_COMFORT'], # Use string literal for key and standardized name
        'K_HUMIDITY_COMFORT': normalize_k['K_HUMIDITY_COMFORT'],     # Use string literal for key and standardized name
        'K_ENERGY_CONSUMPTION': normalize_k['K_ENERGY_CONSUMPTION'] # Use string literal for key
    }
  else:
    print(f"Unknown method: {method}. Using default.")
    return

  # Update system parameter
  for key, value in new_k_values.items():
    system_parameters[key] = value
  print("K values updated successfully.")


# -------------------------
# Reward Weighting System
# -------------------------
class RewardWeighter:
    def __init__(self, num_components=5, lr=0.05):
        self.num_components = num_components
        self.lr = lr
        self.weights = np.ones(num_components) / num_components
        self.running_variance = np.zeros(num_components)
        self.component_history = {i: [] for i in range(num_components)}

    def update(self, reward_components):
        reward_components = np.array(reward_components)

        # Record historical values ​​(for analysis)
        for i, value in enumerate(reward_components):
            self.component_history[i].append(value)

        # Track magnitude / variance (importance)
        self.running_variance = 0.9 * self.running_variance + 0.1 * (reward_components ** 2)

        # Convert variance to raw importance ratio
        raw_importance = self.running_variance / (np.sum(self.running_variance) + 1e-9)

        # Smooth update
        self.weights = (1 - self.lr) * self.weights + self.lr * raw_importance

        # Normalize so sum = 1
        self.weights = self.weights / np.sum(self.weights)

    def get_weighted_reward(self, reward_components):
        return float(np.dot(self.weights, reward_components))

    def get_component_stats(self):
        """Get statistics for each reward component"""
        stats = {}
        for i, values in self.component_history.items():
          if values:
            stats[f'component_{i}'] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        return stats


# -------------------------
# Reward Functions
# -------------------------
def calculate_co2_reward(current_co2_ppm):

    co2_min = system_parameters['THRESHOLD_CO2_PPM_MIN']      # 800 ppm
    co2_mid = system_parameters['THRESHOLD_CO2_PPM_MID']      # 1000 ppm
    co2_max = system_parameters['THRESHOLD_CO2_PPM_MAX']      # 1500 ppm
    k_co2 = system_parameters['K_CO2']

    # Region 1: CO2 ≤ 800 → reward = +1
    if current_co2_ppm <= co2_min:
        return 1.0

    # Region 2: 800 < CO2 ≤ 1000 → small positive reward depending on distance from 1000
    if co2_min < current_co2_ppm <= co2_mid:
        return k_co2 * (1000 - current_co2_ppm)**1.5

    # Region 3: 1000 < CO2 < 1500 → negative quadratic penalty
    if co2_mid < current_co2_ppm < co2_max:
        return -k_co2 * (current_co2_ppm - 1000)**1.5

    # Region 4: CO2 ≥ 1500 → hard penalty
    return -1.0

def calculate_nh3_reward(current_nh3_ppm):

    k_nh3 = system_parameters.get('K_NH3', 0.05)
    thr_low = system_parameters.get('NH3_THRESHOLD_PPM_MIN', 1.6)
    thr_high = system_parameters.get('NH3_THRESHOLD_PPM_MAX', 3.0)

    # Safe clipping (in case sensor reading is weird)
    nh3 = float(np.clip(current_nh3_ppm, 0.0, 1e6))

    if nh3 < thr_low:
        return 1.0
    elif thr_low <= nh3 < thr_high:
        return -k_nh3 * (nh3 - thr_low) ** 1.5
    else:  # nh3 >= thr_high
        return -1.0

def calculate_h2s_reward(current_h2s_ppm):
    k_h2s = system_parameters['K_H2S']
    thr_low = system_parameters['H2S_THRESHOLD_PPM_MIN']   # 0.01 ppm
    thr_high = system_parameters['H2S_THRESHOLD_PPM_MAX']  # 1.5 ppm

    h2s = float(np.clip(current_h2s_ppm, 0.0, 1e6))

    # Region 1: Good air quality
    if h2s < thr_low:
        return 1.0

    # Region 2: Mild pollution (soft penalty)
    elif thr_low <= h2s < thr_high:
        return -k_h2s * (h2s - thr_low) ** 1.5

    # Region 3: Dangerous (hard penalty)
    else:
        return -1.0

def calculate_comfort_reward(current_temperature_c, current_humidity_percent):
    temp_low = system_parameters['THRESHOLD_TEMPERATURE_C']['LOW']
    temp_high = system_parameters['THRESHOLD_TEMPERATURE_C']['HIGH']
    hum_low = system_parameters['THRESHOLD_HUMIDITY_PERCENT']['LOW']
    hum_high = system_parameters['THRESHOLD_HUMIDITY_PERCENT']['HIGH']
    k_temp_comfort = system_parameters['K_TEMPERATURE_COMFORT'] # Standardized key
    k_hum_comfort = system_parameters['K_HUMIDITY_COMFORT']     # Standardized key

    # Temperature reward
    if current_temperature_c <= temp_low:
        r_temp = 1.0
    elif temp_low < current_temperature_c < temp_high:
        r_temp = -k_temp_comfort * ((current_temperature_c - temp_low) ** 1.5)
    else:
        r_temp = -1.0

    # Humidity reward
    if current_humidity_percent <= hum_low:
        r_hum = 1.0
    elif hum_low < current_humidity_percent < hum_high:
        r_hum = -k_hum_comfort * ((current_humidity_percent - hum_low) ** 1.5)
    else:
        r_hum = -1.0

    # Combine
    total_comfort_penalty = (r_temp + r_hum) / 2.0

    return total_comfort_penalty

def calculate_energy_cost(action):
    power_exhaust_fan = system_parameters['POWER_EXHAUST_FAN_W']
    power_ceiling_fan = system_parameters['POWER_CEILING_FAN_W']
    power_dehumidifier = system_parameters['POWER_DEHUMIDIFIER_W']
    k_energy = system_parameters['K_ENERGY_CONSUMPTION']
    delta_t_hours = system_parameters['DELTA_T_HOURS']

    total_power_consumption = 0
    if action.get('exhaust_fan', 0) == 1:
        total_power_consumption += power_exhaust_fan
    if action.get('ceiling_fan', 0) == 1:
        total_power_consumption += power_ceiling_fan
    if action.get('dehumidifier', 0) == 1:
        total_power_consumption += power_dehumidifier

    et = total_power_consumption * delta_t_hours

    energy_cost = -k_energy * et
    return energy_cost

def calculate_total_reward(state, action, weighter=None):
    co2_r = calculate_co2_reward(state['co2_ppm'])
    nh3_r = calculate_nh3_reward(state['nh3_ppm'])
    h2s_r = calculate_h2s_reward(state['h2s_ppm'])
    comfort_r = calculate_comfort_reward(state['temperature_c'], state['humidity_percent'])
    energy_c = calculate_energy_cost(action)

    reward_components = [co2_r, nh3_r, h2s_r, comfort_r, energy_c]

    if weighter is None:
        # fallback: simple sum
        return sum(reward_components), reward_components

    # Weighted reward
    weighted_reward = weighter.get_weighted_reward(reward_components)

    return weighted_reward, reward_components


# -------------------------
# Public_Toilet State
# -------------------------
class Public_Toilet_State:
    def __init__(self):
        # Initialize environmental state variables with random values within reasonable ranges
        self.co2_ppm = np.random.uniform(400.0, 1501.0)  # CO2 level in parts per million (ppm)
        self.nh3_ppm = np.random.uniform(0.032, 5.0)    # NH3 level in ppm
        self.h2s_ppm = np.random.uniform(0.00011, 2.1)   # H2S level in ppm
        self.temperature_c = np.random.uniform(25.5, 31.0) # Temperature in degrees Celsius
        self.humidity_percent = np.random.uniform(49.3, 74.0) # Relative humidity in percentage

        # Equipment status (boolean: ON/OFF)
        self.exhaust_fan_on = False
        self.ceiling_fan_on = False
        self.dehumidifier_on = False

    # Apply action + simulate environment
    def update_state(self, action):
        # Update equipment status
        self.exhaust_fan_on = bool(action['exhaust_fan'])
        self.ceiling_fan_on = bool(action['ceiling_fan'])
        self.dehumidifier_on = bool(action['dehumidifier'])

        # Toilet details
        room_volume = 17.64 * 3.0  # 52.92 m³
        time_step_hours = 1/60

        # Set ACH value according to exhaust fan status
        # Reference value:
        # Off: 0.5 ACH (natural ventilation)
        # On: 6 ACH (ASHRAE recommended minimum ventilation rate for sanitation facilities)
        if not self.exhaust_fan_on:
          ach = 0.5  # Natural ventilation, low air exchange rate
        elif self.exhaust_fan_on:
          ach = 6.0

        # Calculate the air exchange rate (m³/h)
        air_exchange_rate = ach * room_volume  # m³/h

        # Calculate the air exchange rate (the proportion removed by exhaust air) within the time step
        air_removed_ratio = air_exchange_rate * time_step_hours / room_volume
        air_removed_ratio = min(air_removed_ratio, 1.0)  # Ensure it does not exceed 100%

        # Initialize generated pollutant amounts
        co2_generated = 0.0
        nh3_generated = 0.0
        h2s_generated = 0.0

        # Generate baseline pollutant amounts
        # These are constant background sources
        co2_generated += np.random.uniform(1, 5)
        nh3_generated += np.random.uniform(0.01, 0.05)
        h2s_generated += np.random.uniform(0.001, 0.01)


        # # Exhaust fan decreases pollutants directly (removed dependency on ACH)
        # if self.exhaust_fan_on:
        #     self.co2_ppm -= np.random.uniform(5, 15) # Larger reduction if exhaust fan is on
        #     self.nh3_ppm -= np.random.uniform(0.05, 0.2)
        #     self.h2s_ppm -= np.random.uniform(0.005, 0.02)
        # else:
        #     self.co2_ppm -= np.random.uniform(0.5, 5) # Smaller natural reduction
        #     self.nh3_ppm -= np.random.uniform(0.001, 0.05)
        #     self.h2s_ppm -= np.random.uniform(0.0001, 0.005)

        # update CO2
        self.co2_ppm = self.co2_ppm * (1 - air_removed_ratio) + co2_generated

        # NH3 update (considering natural decay）
        natural_decay_nh3 = 0.02  # decays naturally by 2% per minute.
        self.nh3_ppm = self.nh3_ppm * (1 - air_removed_ratio) * (1 - natural_decay_nh3) + nh3_generated

        # H2S update (considering natural degradation)
        natural_decay_h2s = 0.03  # decays naturally by 3% per minute.
        self.h2s_ppm = self.h2s_ppm * (1 - air_removed_ratio) * (1 - natural_decay_h2s) + h2s_generated

        # Ensure pollutants don't go below zero
        self.co2_ppm = np.clip(self.co2_ppm, 300, 2000)
        self.nh3_ppm = np.clip(self.nh3_ppm, 0.0, 10.0)
        self.h2s_ppm = np.clip(self.h2s_ppm, 0.0, 5.0)

        # Simulate temperature changes
        # Natural fluctuation
        self.temperature_c += np.random.uniform(-0.1, 0.3)
        # Ceiling fan effect (cooling)
        if self.ceiling_fan_on:
            self.temperature_c -= np.random.uniform(0.3, 0.7)

        # Simulate humidity changes
        # Natural fluctuation
        self.humidity_percent += np.random.uniform(-0.1, 0.3)
        # Dehumidifier effect (decreasing humidity)
        if self.dehumidifier_on:
            self.humidity_percent -= np.random.uniform(0.5, 1.0)

        # Clip environmental values to reasonable ranges
        self.temperature_c = np.clip(self.temperature_c, 22, 34)
        self.humidity_percent = np.clip(self.humidity_percent, 45, 80)

    # --------------------------------
    # State for RL
    # --------------------------------
    def get_current_state(self):
        # Returns a dictionary of current state variables
        return {
            "co2_ppm": self.co2_ppm,
            "nh3_ppm": self.nh3_ppm,
            "h2s_ppm": self.h2s_ppm,
            "temperature_c": self.temperature_c,
            "humidity_percent": self.humidity_percent,
        }

    def get_current_state_array(self):
        """Returns state as numpy array in the order: NH3, H2S, CO2, Temperature, Humidity"""
        return np.array([
            self.nh3_ppm,
            self.h2s_ppm,
            self.co2_ppm,
            self.temperature_c,
            self.humidity_percent
        ], dtype=np.float32)

    
# -------------------------
# ACTION SPACE
# -------------------------
device_states = [0, 1]
combo = itertools.product(device_states, repeat=3)
ACTION_SPACE = [
    {'exhaust_fan': c[0], 'ceiling_fan': c[1], 'dehumidifier': c[2]}
    for c in combo
]


# -------------------------
# Public_Toilet Environment
# -------------------------
class ToiletEnv(gym.Env): # Inherit from gymnasium.Env
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    reward_range = (-float('inf'), float('inf'))

    def __init__(self, episode_length=1440, normalize_states=False, seed=42, k_tuning_method=None, silent=False):
        super().__init__()

        self.episode_length = episode_length
        self.normalize_states = normalize_states
        self.seed = seed
        self.k_tuning_method = k_tuning_method
        self.silent = silent # Add silent parameter

        # If a K-value tuning method is specified, update the K-value.
        if k_tuning_method and not self.silent: # Conditionally print during init
          print(f"\nApplying K-value tuning method: {k_tuning_method}")
          update_k_values(k_tuning_method)
          print("Updated K-values:")
          # Use standardized key names for printing
          for key in ['K_CO2', 'K_NH3', 'K_H2S', 'K_TEMPERATURE_COMFORT', 'K_HUMIDITY_COMFORT', 'K_ENERGY_CONSUMPTION']:
            print(f"{key}: {system_parameters[key]:.6f}")
        elif k_tuning_method:
            update_k_values(k_tuning_method) # Update even if silent

        self.reward_weighter = RewardWeighter(num_components=5)

        # Set random seeds
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Action space
        self.action_space_list = ACTION_SPACE
        self.action_space_size = len(self.action_space_list)

        # Define action_space for Gymnasium compatibility
        self.action_space = spaces.Discrete(self.action_space_size)

        # State dimensions: [NH3, H2S, CO2, Temperature, Humidity]
        self.state_dim = 5

        # State standardization range
        self.state_ranges = {
            'co2_ppm': (300.0, 2000.0),
            'nh3_ppm': (0.0, 10.0),
            'h2s_ppm': (0.0, 5.0),
            'temperature_c': (22.0, 34.0),
            'humidity_percent': (45.0, 80.0)
        }

        # Define observation_space for Gymnasium compatibility
        if normalize_states:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
            )
        else:
            # These values need to be adjusted to match your actual state ranges
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 300.0, 22.0, 45.0], dtype=np.float32),
                high=np.array([10.0, 5.0, 2000.0, 34.0, 80.0], dtype=np.float32),
                dtype=np.float32
            )

        self.Public_Toilet_state = None
        self.current_step = 0

        # reset
        self.reset()

    def _normalize_state(self, state_dict):
        """Normalize state values to [0, 1] range"""
        normalized = []

        # NH3
        nh3_min, nh3_max = self.state_ranges['nh3_ppm']
        nh3_norm = (state_dict['nh3_ppm'] - nh3_min) / (nh3_max - nh3_min)
        normalized.append(np.clip(nh3_norm, 0.0, 1.0))

        # H2S
        h2s_min, h2s_max = self.state_ranges['h2s_ppm']
        h2s_norm = (state_dict['h2s_ppm'] - h2s_min) / (h2s_max - h2s_min)
        normalized.append(np.clip(h2s_norm, 0.0, 1.0))

        # CO2
        co2_min, co2_max = self.state_ranges['co2_ppm']
        co2_norm = (state_dict['co2_ppm'] - co2_min) / (co2_max - co2_min)
        normalized.append(np.clip(co2_norm, 0.0, 1.0))

        # Temperature
        temp_min, temp_max = self.state_ranges['temperature_c']
        temp_norm = (state_dict['temperature_c'] - temp_min) / (temp_max - temp_min)
        normalized.append(np.clip(temp_norm, 0.0, 1.0))

        # Humidity
        hum_min, hum_max = self.state_ranges['humidity_percent']
        hum_norm = (state_dict['humidity_percent'] - hum_min) / (hum_max - hum_min)
        normalized.append(np.clip(hum_norm, 0.0, 1.0))

        return np.array(normalized, dtype=np.float32)

    # Reset mechanism
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed) # Use env-level seed for initial state randomization
            random.seed(seed)

        self.Public_Toilet_state = Public_Toilet_State()
        self.current_step = 0

        observation = (self._normalize_state(self.Public_Toilet_state.get_current_state()) if self.normalize_states
                      else self.Public_Toilet_state.get_current_state_array())
        info = {'reset_info': 'Environment reset successfully'}

        return observation, info

    # step mechanism
    def step(self, action_idx):
    # def step(self, action):

        # Get action from index
        action = self.action_space_list[action_idx]

        # Update the Public_Toilet_state based on the action
        self.Public_Toilet_state.update_state(action)

        # Get the new state
        current_state_dict = self.Public_Toilet_state.get_current_state()

        if self.normalize_states:
            observation = self._normalize_state(current_state_dict)
        else:
            observation = self.Public_Toilet_state.get_current_state_array()

        # Calculate the reward using the dedicated function
        reward, components = calculate_total_reward(current_state_dict, action, self.reward_weighter)

        info = {
            'step': self.current_step,
            'action': action,
            'state_dict': current_state_dict,
            'equipment': {
                'exhaust_fan': self.Public_Toilet_state.exhaust_fan_on,
                'ceiling_fan': self.Public_Toilet_state.ceiling_fan_on,
                'dehumidifier': self.Public_Toilet_state.dehumidifier_on,
        },
        'reward_components': {
            'co2': components[0],
            'nh3': components[1],
            'h2s': components[2],
            'comfort': components[3],
            'energy': components[4],
        }
    }


        # Update reward weights
        self.reward_weighter.update(components)

        # Print weights (so you can see)
        if not self.silent:
            # print("Reward components:", components) # Commented out for cleaner output during training
            # print("Current weights:", self.reward_weighter.weights) # Commented out for cleaner output during training
            pass

        # Increment current step
        self.current_step += 1

        # Determine if the episode is done (terminated or truncated)
        terminated = False
        truncated = False

        # Safety termination (e.g., extreme pollutant levels)
        if (current_state_dict['co2_ppm'] > 2000 or
            current_state_dict['nh3_ppm'] > 10 or
            current_state_dict['h2s_ppm'] > 5):
            terminated = True
        # Episode limit
        if self.current_step >= self.episode_length:
            truncated = True

        info = {
            'step': self.current_step,
            'action': action,
            'state_dict': current_state_dict,
            'equipment': {
                'exhaust_fan': self.Public_Toilet_state.exhaust_fan_on,
                'ceiling_fan': self.Public_Toilet_state.ceiling_fan_on,
                'dehumidifier': self.Public_Toilet_state.dehumidifier_on,
            }
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Display current state"""
        state_dict = self.Public_Toilet_state.get_current_state()
        print(f"\nStep {self.current_step}:")
        print(f"  NH3: {state_dict['nh3_ppm']:.6f} ppm")
        print(f"  H2S: {state_dict['h2s_ppm']:.6f} ppm")
        print(f"  CO2: {state_dict['co2_ppm']:.6f} ppm")
        print(f"  Temp: {state_dict['temperature_c']:.1f}°C")
        print(f"  Hum: {state_dict['humidity_percent']:.1f}%")
        print(f"  Equipment: Exhaust={self.Public_Toilet_state.exhaust_fan_on}, "
              f"Ceiling={self.Public_Toilet_state.ceiling_fan_on}, "
              f"Dehum={self.Public_Toilet_state.dehumidifier_on}")
    def get_reward_stats(self):
        """Get reward statistics"""
        return self.reward_weighter.get_component_stats()

    def close(self):
        """Clean up environment"""
        pass


# -------------------------
# Wrapper
# -------------------------
class PublicToiletEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # The original action space is a dict, which we flatten into MultiDiscrete([2,2,2]).
        # For gymnasium compatibility, ensure the action space is defined using gymnasium.spaces
        self.action_space = spaces.MultiDiscrete([2, 2, 2])

        # The original observation space is a dict, which we flatten into a Box.
        # For gymnasium compatibility, ensure the observation space is defined using gymnasium.spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),  # nh3, h2s, co2, temp, hum
            dtype=np.float32
        )

    def _flatten_observation(self, obs):
        """Flatten the dictionary state into a vector"""
        # This method is not used directly by the PublicToiletEnvWrapper.step or reset anymore
        # as the wrapped ToiletEnv now returns an array directly.
        # However, keeping it for clarity if the underlying env ever changes its output.
        if isinstance(obs, dict): # If the obs is a dict, flatten it
            flat_obs = np.array([
                obs['nh3_ppm'],
                obs['h2s_ppm'],
                obs['co2_ppm'],
                obs['temperature_c'],
                obs['humidity_percent']
            ], dtype=np.float32)
            return flat_obs
        return obs # Otherwise, assume it's already flat (e.g., from normalize_states)

    # Helper to find the integer index for a given action dict
    def _get_action_index(self, action_dict):
        for idx, env_action_dict in enumerate(self.env.action_space_list):
            if env_action_dict == action_dict:
                return idx
        raise ValueError(f"Action dictionary {action_dict} not found in environment's action space list.")

    def step(self, action):
        # MultiDiscrete action (e.g., np.array([0, 1, 0])) -> dict
        # Ensure 'action' is treated as a 1D array if it comes as (1, N)
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.flatten()

        action_dict = {
            'exhaust_fan': int(action[0]),
            'ceiling_fan': int(action[1]),
            'dehumidifier': int(action[2])
        }

        # Get the integer index for the underlying ToiletEnv
        action_idx = self._get_action_index(action_dict)

        # The wrapped environment (ToiletEnv) now returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        # The obs from the wrapped env is already an array, no need to flatten again if normalize_states is true
        # If normalize_states is false, it's also an array.
        # flat_obs = self._flatten_observation(obs) # This line might be redundant if ToiletEnv is properly configured
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # The obs from the wrapped env is already an array.
        # flat_obs = self._flatten_observation(obs) # This line might be redundant if ToiletEnv is properly configured
        return obs, info
    

# -------------------------
# Evaluate Policy Metrics
# ------------------------- 
def evaluate_policy_metrics(model, env, n_episodes=300, window=30):
    episode_rewards = []

    pollutant_ep = []
    comfort_ep   = []   # NEW
    energy_ep    = []   # NEW

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        total_reward = 0
        pollutant_sum = 0
        comfort_sum   = 0   # NEW
        energy_sum    = 0   # NEW

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

            ri = info.get('reward_components', {
                    "co2": 0,
                    "nh3": 0,
                    "h2s": 0,
                    "comfort": 0,
                    "energy": 0
            })

            # ---- reward components ----
            pollutant_sum += ri["co2"] + ri["nh3"] + ri["h2s"]
            comfort_sum   += ri["comfort"]
            energy_sum    += ri["energy"]

        episode_rewards.append(total_reward)
        pollutant_ep.append(pollutant_sum)
        comfort_ep.append(comfort_sum)
        energy_ep.append(energy_sum)

    rewards = np.array(episode_rewards)

    return {
        # ---- overall ----
        "avg_episode_reward": np.mean(rewards),
        "cumulative_reward": np.sum(rewards),
        "reward_variance": np.var(rewards[-window:]) if len(rewards) >= window else np.var(rewards),

        # ---- decomposed averages ----
        "avg_pollutant_reward": np.mean(pollutant_ep),
        "avg_comfort_reward":   np.mean(comfort_ep),
        "avg_energy_reward":    np.mean(energy_ep),

        "all_rewards": rewards
    }

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Public Toilet PPO Evaluation", layout="wide")
st.title("🚻 PPO Evaluation: Public Toilet")

# -----------------------------
# Sidebar
# -----------------------------
num_episodes = st.sidebar.number_input("Number of Episodes", value=20, min_value=1, step=1)
episode_length = st.sidebar.number_input("Episode Length", value=1440, step=1)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
speed = st.sidebar.slider("Speed (seconds per episode)", min_value=0.01, max_value=1.0, value=1.0, step=0.1)

# -----------------------------
# Control Buttons
# -----------------------------
play_button = st.sidebar.button("▶️ Play / Resume")
pause_button = st.sidebar.button("⏸ Pause")
replay_button = st.sidebar.button("🔁 Replay")

# -----------------------------
# Session State
# -----------------------------
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "current_episode" not in st.session_state:
    st.session_state.current_episode = 1
if "episode_rewards" not in st.session_state:
    st.session_state.episode_rewards = []
if "episode_numbers" not in st.session_state:
    st.session_state.episode_numbers = []
if "pollutants_rewards" not in st.session_state:
    st.session_state.pollutants_rewards = []
if "comfort_rewards" not in st.session_state:
    st.session_state.comfort_rewards = []
if "energy_rewards" not in st.session_state:
    st.session_state.energy_rewards = []
if "reward_components_history" not in st.session_state:
    st.session_state.reward_components_history = []

# -----------------------------
# Load PPO Model
# -----------------------------
@st.cache_resource
def load_ppo_model():
    env_tmp = ToiletEnv(episode_length=episode_length, normalize_states=True, seed=seed)
    env_tmp = PublicToiletEnvWrapper(env_tmp)
    model = PPO.load("trained_models/ppo_model.zip", env=env_tmp)
    return model

# -----------------------------
# Create environment
# -----------------------------
@st.cache_resource
def make_env():
    env = ToiletEnv(episode_length=episode_length, normalize_states=True, seed=seed)
    env = PublicToiletEnvWrapper(env)
    return Monitor(env)

env = make_env()
model = load_ppo_model()

# -----------------------------
# Placeholders
# -----------------------------
state_placeholder = st.empty()
fig_placeholder = st.empty()
table_placeholder = st.empty()

# -----------------------------
# Create subplot figure with 2x2 layout
# -----------------------------
fig = sp.make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Total Reward per Episode",
        "Pollutants Reward (NH3+H2S+CO2) per Episode",
        "Comfort Reward (Temperature+Humidity) per Episode",
        "Energy Reward per Episode"
    ),
    vertical_spacing=0.15,
    horizontal_spacing=0.15
)

# Add traces for each subplot
# Top-left: Total Reward
fig.add_trace(
    go.Scatter(x=[], y=[], mode='lines+markers', line=dict(color='blue', width=2), name='Total Reward'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=[1,1], y=[-1000,1000], mode='lines', line=dict(color='white', dash='dash'), name='Current Episode', showlegend=False),
    row=1, col=1
)

# Top-right: Pollutants Reward
fig.add_trace(
    go.Scatter(x=[], y=[], mode='lines+markers', line=dict(color='red', width=2), name='Pollutants Reward'),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=[1,1], y=[-1000,1000], mode='lines', line=dict(color='white', dash='dash'), name='Current Episode', showlegend=False),
    row=1, col=2
)

# Bottom-left: Comfort Reward
fig.add_trace(
    go.Scatter(x=[], y=[], mode='lines+markers', line=dict(color='green', width=2), name='Comfort Reward'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=[1,1], y=[-1000,1000], mode='lines', line=dict(color='white', dash='dash'), name='Current Episode', showlegend=False),
    row=2, col=1
)

# Bottom-right: Energy Reward
fig.add_trace(
    go.Scatter(x=[], y=[], mode='lines+markers', line=dict(color='orange', width=2), name='Energy Reward'),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=[1,1], y=[-1000,1000], mode='lines', line=dict(color='white', dash='dash'), name='Current Episode', showlegend=False),
    row=2, col=2
)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    legend=dict(x=1.02, y=1)
)

# Update axis labels
fig.update_xaxes(title_text="Episode", row=1, col=1)
fig.update_yaxes(title_text="Total Reward", row=1, col=1)
fig.update_xaxes(title_text="Episode", row=1, col=2)
fig.update_yaxes(title_text="Pollutants Reward", row=1, col=2)
fig.update_xaxes(title_text="Episode", row=2, col=1)
fig.update_yaxes(title_text="Comfort Reward", row=2, col=1)
fig.update_xaxes(title_text="Episode", row=2, col=2)
fig.update_yaxes(title_text="Energy Reward", row=2, col=2)

# Set ranges - only x-axis is fixed
for row in [1, 2]:
    for col in [1, 2]:
        fig.update_xaxes(range=[0, num_episodes + 1], row=row, col=col)

# -----------------------------
# Handle button clicks
# -----------------------------
if play_button:
    st.session_state.is_playing = True

if pause_button:
    st.session_state.is_playing = False

if replay_button:
    st.session_state.is_playing = True
    st.session_state.current_episode = 1
    st.session_state.episode_rewards = []
    st.session_state.episode_numbers = []
    st.session_state.pollutants_rewards = []
    st.session_state.comfort_rewards = []
    st.session_state.energy_rewards = []
    st.session_state.reward_components_history = []

# -----------------------------
# Run Evaluation Loop (Safe Version)
# -----------------------------
while st.session_state.current_episode <= num_episodes:
    if not st.session_state.is_playing:
        break

    ep = st.session_state.current_episode

    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, _ = reset_result
    else:
        obs = reset_result
    done = False

    # Initialize episode rewards
    total_reward = 0.0
    episode_pollutants_reward = 0.0
    episode_comfort_reward = 0.0
    episode_energy_reward = 0.0
    step_count = 0

    while not done:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Safely get reward components
        ri = getattr(env.unwrapped, 'reward_info', {"co2": 0, "nh3": 0, "h2s": 0, "comfort": 0, "energy": 0})
        pollutants_reward_step = ri.get("co2", 0) + ri.get("nh3", 0) + ri.get("h2s", 0)
        comfort_reward_step = ri.get("comfort", 0)
        energy_reward_step = ri.get("energy", 0)

        # Accumulate rewards
        total_reward += reward
        episode_pollutants_reward += pollutants_reward_step
        episode_comfort_reward += comfort_reward_step
        episode_energy_reward += energy_reward_step

        done = terminated or truncated
        step_count += 1

    # Calculate average component rewards for this episode
    if step_count > 0:
        avg_co2 = episode_pollutants_reward / 3 / step_count
        avg_nh3 = episode_pollutants_reward / 3 / step_count
        avg_h2s = episode_pollutants_reward / 3 / step_count
        avg_comfort = episode_comfort_reward / step_count
        avg_energy = episode_energy_reward / step_count
    else:
        avg_co2 = avg_nh3 = avg_h2s = avg_comfort = avg_energy = 0.0

    # Save episode data
    st.session_state.episode_rewards.append(total_reward)
    st.session_state.episode_numbers.append(ep)
    st.session_state.pollutants_rewards.append(episode_pollutants_reward)
    st.session_state.comfort_rewards.append(episode_comfort_reward)
    st.session_state.energy_rewards.append(episode_energy_reward)

    # Store individual component rewards for table
    st.session_state.reward_components_history.append({
        'Episode': ep,
        'NH3_reward': avg_nh3,
        'H2S_reward': avg_h2s,
        'CO2_reward': avg_co2,
        'Comfort_reward': avg_comfort,
        'Energy_reward': avg_energy,
        'Total_reward': total_reward
    })

    # Safely get final state info
    state_dict = info.get('state_dict', {"nh3_ppm": 0, "h2s_ppm": 0, "co2_ppm": 0, "temperature_c": 0, "humidity_percent": 0})

    # Update state display
    state_placeholder.markdown(
        f"**Episode {ep} finished!**  \n"
        f"**Total Reward:** {total_reward:.2f}  \n"
        f"**Pollutants Reward:** {episode_pollutants_reward:.2f}  \n"
        f"**Comfort Reward:** {episode_comfort_reward:.2f}  \n"
        f"**Energy Reward:** {episode_energy_reward:.2f}  \n"
        f"**Final NH3:** {state_dict['nh3_ppm']:.3f} ppm  \n"
        f"**Final H2S:** {state_dict['h2s_ppm']:.3f} ppm  \n"
        f"**Final CO2:** {state_dict['co2_ppm']:.0f} ppm  \n"
        f"**Final Temp:** {state_dict['temperature_c']:.1f} °C  \n"
        f"**Final Humidity:** {state_dict['humidity_percent']:.1f} %"
    )

    # -----------------------------
    # Update all plots
    # -----------------------------
    # Total reward plot
    fig.data[0].x = st.session_state.episode_numbers
    fig.data[0].y = st.session_state.episode_rewards
    y_min_total = min(st.session_state.episode_rewards, default=0)
    y_max_total = max(st.session_state.episode_rewards, default=100)
    y_padding_total = (y_max_total - y_min_total) * 0.1 + 1
    fig.data[1].x = [ep, ep]
    fig.data[1].y = [y_min_total - y_padding_total, y_max_total + y_padding_total]
    fig.update_yaxes(range=[y_min_total - y_padding_total, y_max_total + y_padding_total], row=1, col=1)

    # Pollutants reward plot
    fig.data[2].x = st.session_state.episode_numbers
    fig.data[2].y = st.session_state.pollutants_rewards
    y_min_poll = min(st.session_state.pollutants_rewards, default=0)
    y_max_poll = max(st.session_state.pollutants_rewards, default=100)
    y_padding_poll = (y_max_poll - y_min_poll) * 0.1 + 1
    fig.data[3].x = [ep, ep]
    fig.data[3].y = [y_min_poll - y_padding_poll, y_max_poll + y_padding_poll]
    fig.update_yaxes(range=[y_min_poll - y_padding_poll, y_max_poll + y_padding_poll], row=1, col=2)

    # Comfort reward plot
    fig.data[4].x = st.session_state.episode_numbers
    fig.data[4].y = st.session_state.comfort_rewards
    y_min_comfort = min(st.session_state.comfort_rewards, default=0)
    y_max_comfort = max(st.session_state.comfort_rewards, default=100)
    y_padding_comfort = (y_max_comfort - y_min_comfort) * 0.1 + 1
    fig.data[5].x = [ep, ep]
    fig.data[5].y = [y_min_comfort - y_padding_comfort, y_max_comfort + y_padding_comfort]
    fig.update_yaxes(range=[y_min_comfort - y_padding_comfort, y_max_comfort + y_padding_comfort], row=2, col=1)

    # Energy reward plot
    fig.data[6].x = st.session_state.episode_numbers
    fig.data[6].y = st.session_state.energy_rewards
    y_min_energy = min(st.session_state.energy_rewards, default=0)
    y_max_energy = max(st.session_state.energy_rewards, default=100)
    y_padding_energy = (y_max_energy - y_min_energy) * 0.1 + 1
    fig.data[7].x = [ep, ep]
    fig.data[7].y = [y_min_energy - y_padding_energy, y_max_energy + y_padding_energy]
    fig.update_yaxes(range=[y_min_energy - y_padding_energy, y_max_energy + y_padding_energy], row=2, col=2)

    fig_placeholder.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Update reward components table
    # -----------------------------
    if st.session_state.reward_components_history:
        df = pd.DataFrame(st.session_state.reward_components_history)
        display_df = df.copy()
        for col in ['NH3_reward','H2S_reward','CO2_reward','Comfort_reward','Energy_reward']:
            display_df[col] = display_df[col].map('{:,.4f}'.format)
        display_df['Total_reward'] = display_df['Total_reward'].map('{:,.2f}'.format)
        display_df = display_df[['Episode','NH3_reward','H2S_reward','CO2_reward','Comfort_reward','Energy_reward','Total_reward']]

        table_placeholder.markdown("### Episode Reward Components")
        table_placeholder.markdown("""
        <style>
        .dataframe { font-size: 12px; }
        .dataframe thead th { background-color: #f0f0f0; text-align: center; }
        .dataframe tbody tr:nth-child(even) { background-color: #f9f9f9; }
        </style>
        """, unsafe_allow_html=True)
        table_placeholder.dataframe(display_df, use_container_width=True, height=400, hide_index=True)

    # Increment episode
    st.session_state.current_episode += 1
    time.sleep(speed)

st.success("✅ PPO Evaluation Finished!")
