import streamlit as st
import numpy as np

from stable_baselines3 import PPO, DQN, SAC, A2C
from stable_baselines3.common.monitor import Monitor
import itertools
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import plotly.graph_objects as go



# -------------------------
# Configuration
# -------------------------


system_parameters = {
    # Power ratings in Watts
    'POWER_EXHAUST_FAN_W': 55,
    'POWER_CEILING_FAN_W': 42.5,
    'POWER_DEHUMIDIFIER_W': 500,

    # Threshold values for environmental parameters
    'THRESHOLD_CO2_PPM_MIN': 400,
    'THRESHOLD_CO2_PPM1': 800,
    'THRESHOLD_CO2_PPM2': 1000,
    'THRESHOLD_CO2_PPM_MAX': 1500,

    # NH3 Thresholds
    'NH3_THRESHOLD_PPM_MIN': 0.032,
    'NH3_THRESHOLD_PPM1': 3,
    'NH3_THRESHOLD_PPM_MAX': 5,

    # H2S Thresholds
    'H2S_THRESHOLD_PPM1': 0.01,
    'H2S_THRESHOLD_PPM2': 1.5,

    'THRESHOLD_TEMPERATURE_C': {
        'LOW': 28,              # Degrees Celsius
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
      'CO2': 1000,
      'NH3': 10,
      'H2S': 0.5,
      'Temperature': 5,
      'Humidity': 20,
      'Energy':100
  }

  target_range = 1.0
  k_values = {}

  #Normalization of secondary penalties
  k_values['K_CO2'] = target_range / (benchmarks['CO2'] ** 2)
  k_values['K_NH3'] = target_range / (benchmarks['NH3'] ** 2)
  k_values['K_H2S'] = target_range / (benchmarks['H2S'] ** 2)
  # Standardize key names for consistency
  k_values['K_TEMPERATURE_COMFORT'] = target_range / (benchmarks['Temperature'] ** 2)
  k_values['K_HUMIDITY_COMFORT'] = target_range / (benchmarks['Humidity'] ** 2)
  k_values['K_ENERGY_CONSUMPTION'] = target_range / (benchmarks['Energy'] ** 2)

  return k_values

def set_k_by_priority():
  """K value is set based on priority
  Priority 1-5, 5 being the highest"""

  priorities = {
      'CO2': 3,
      'NH3': 5,
      'H2S': 5,
      'TEMPERATURE_COMFORT': 2,
      'HUMIDITY_COMFORT': 3,
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

    co2_min = system_parameters['THRESHOLD_CO2_PPM1']      # 800 ppm
    co2_mid = system_parameters['THRESHOLD_CO2_PPM2']      # 1000 ppm
    co2_max = system_parameters['THRESHOLD_CO2_PPM_MAX']   # 1500 ppm
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
    thr_low = system_parameters.get('NH3_THRESHOLD_PPM1', 3.0)
    thr_high = system_parameters.get('NH3_THRESHOLD_PPM_MAX', 5.0)

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
    thr_low = system_parameters['H2S_THRESHOLD_PPM1']   # 0.01 ppm
    thr_high = system_parameters['H2S_THRESHOLD_PPM2']  # 1.5 ppm

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
        self.co2_ppm = np.random.uniform(400, 1501)  # CO2 level in parts per million (ppm)
        self.nh3_ppm = np.random.uniform(0.032, 5)    # NH3 level in ppm
        self.h2s_ppm = np.random.uniform(0.00011, 2.1)   # H2S level in ppm
        self.temperature_c = np.random.uniform(25.5, 31) # Temperature in degrees Celsius
        self.humidity_percent = np.random.uniform(49.3, 74) # Relative humidity in percentage

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
        self.co2_ppm = np.clip(self.co2_ppm, 300, 5000)
        self.nh3_ppm = np.clip(self.nh3_ppm, 0.0, 100)
        self.h2s_ppm = np.clip(self.h2s_ppm, 0.0, 20)

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
        self.humidity_percent = np.clip(self.humidity_percent, 55, 85)

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
import gymnasium as gym
from gymnasium import spaces # Import spaces from gymnasium

class ToiletEnv(gym.Env): # Inherit from gymnasium.Env
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    reward_range = (-float('inf'), float('inf'))

    def __init__(self, episode_length=1440, normalize_states=False, seed=None, k_tuning_method=None, silent=False):
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
            'co2_ppm': (300.0, 5000.0),
            'nh3_ppm': (0.0, 5.0),
            'h2s_ppm': (0.0, 5.0),
            'temperature_c': (22.0, 35.0),
            'humidity_percent': (50.0, 90.0)
        }

        # Define observation_space for Gymnasium compatibility
        if normalize_states:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
            )
        else:
            # These values need to be adjusted to match your actual state ranges
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 300.0, 22.0, 55.0], dtype=np.float32),
                high=np.array([100.0, 20.0, 5000.0, 34.0, 85.0], dtype=np.float32),
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
        if (current_state_dict['co2_ppm'] > 5000 or
            current_state_dict['nh3_ppm'] > 30 or
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
        print(f"  NH3: {state_dict['nh3_ppm']:.2f} ppm")
        print(f"  H2S: {state_dict['h2s_ppm']:.3f} ppm")
        print(f"  CO2: {state_dict['co2_ppm']:.0f} ppm")
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

## Wrapper

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
    
def evaluate_policy_metrics(model, env, n_episodes=300, window=300):
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)

    rewards = np.array(episode_rewards)

    # ---- Metrics ----
    avg_reward = np.mean(rewards)                  # average episodic reward
    cumulative_reward = np.sum(rewards)             # cumulative reward

    # Learning stability (variance over last window)
    if len(rewards) >= window:
        stability_variance = np.var(rewards[-window:])
    else:
        stability_variance = np.var(rewards)

    return {
        "avg_episode_reward": avg_reward,
        "cumulative_reward": cumulative_reward,
        "reward_variance": stability_variance,
        "all_rewards": rewards
    }

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Public Toilet PPO Evaluation", layout="wide")
st.title("🚻 Public Toilet PPO Evaluation Dashboard")

# -------------------------
# Sidebar settings
# -------------------------
episode_length = st.sidebar.slider("Episode Length", min_value=240, max_value=1440, step=120, value=1440)
normalize_states = st.sidebar.checkbox("Normalize States", value=True)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
run_button = st.sidebar.button("▶️ Run PPO Evaluation")

# -------------------------
# PPO model path
# -------------------------
MODEL_PATH = "trained_models/ppo_model.zip"

# -------------------------
# Cached environment
# -------------------------
@st.cache_resource
def make_env():
    env = ToiletEnv(episode_length=episode_length, normalize_states=normalize_states, seed=seed, silent=True)
    env = PublicToiletEnvWrapper(env)
    return Monitor(env)

env = make_env()

# -------------------------
# Load PPO model
# -------------------------
@st.cache_resource
def load_ppo_model():
    return PPO.load(MODEL_PATH, env=env)

# -------------------------
# Run Evaluation
# -------------------------
if run_button:
    st.info("Running PPO evaluation...")

    try:
        model = load_ppo_model()
    except Exception as e:
        st.error(f"❌ Failed to load PPO model: {e}")
        st.stop()

    # Placeholders
    reward_placeholder = st.empty()
    state_placeholder = st.empty()
    fig_placeholder = st.empty()

    # Initialize plotting data
    rewards = []
    timesteps = []

    # Reset environment
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    # Plotly figure setup
    fig = go.Figure()
    fig.update_layout(
        title="Cumulative Reward over Time",
        xaxis_title="Timestep",
        yaxis_title="Cumulative Reward",
        xaxis=dict(range=[0, episode_length]),
        yaxis=dict(range=[0, 1]),
    )
    # reward line
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Cumulative Reward'))
    # current step dashed line
    fig.add_trace(go.Scatter(x=[0,0], y=[0,0], mode='lines', line=dict(dash='dash', color='red'), name='Current Step'))

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step = info['step']

        # Append data
        rewards.append(total_reward)
        timesteps.append(step)

        # Update state display
        state_placeholder.markdown(
            f"**Timestep:** {step} / {episode_length}  \n"
            f"**NH3:** {info['state_dict']['nh3_ppm']:.3f} ppm  \n"
            f"**H2S:** {info['state_dict']['h2s_ppm']:.3f} ppm  \n"
            f"**CO2:** {info['state_dict']['co2_ppm']:.0f} ppm  \n"
            f"**Temp:** {info['state_dict']['temperature_c']:.1f} °C  \n"
            f"**Humidity:** {info['state_dict']['humidity_percent']:.1f} %"
        )

        # Update plot
        fig.data[0].x = timesteps
        fig.data[0].y = rewards
        fig.data[1].x = [step, step]
        fig.data[1].y = [0, max(rewards)*1.1]
        fig_placeholder.plotly_chart(fig, use_container_width=True)

        # Wait 1 second for real-time effect
        time.sleep(1)

    st.success(f"🎯 PPO Evaluation Finished | Total Reward = {total_reward:.2f}")