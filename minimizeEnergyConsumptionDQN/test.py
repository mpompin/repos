import os
import numpy as np
import random as rn
import environment
from keras.models import load_model

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#SETTING THE PARAMETERS
number_actions = 5
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5

env = environment.Environment(optimal_temperature= (18.0, 24.0), initial_month =0, initial_number_users = 20, initial_rate_data = 30)
model = load_model('model.h5')

#choose the mode
train = False

#RUNNING ONE YEAR SIMULATION
env.train = False
current_state, _ , _ = env.observe()
for timestep in np.arange(0, 12 * 30 * 24 * 60):
        print(f'{timestep}/{12 * 30 * 24 * 60}')
        q_values = model.predict(current_state.reshape(1,-1))
        action = np.argmax(q_values[0])
        if action - direction_boundary < 0:
                direction = -1
        else:
                direction = 1
        energy_ai = np.abs(action - direction_boundary) * temperature_step
        next_state, reward, game_over = env.update_env(direction=direction, energy_ai=energy_ai,
                                                       month=int(timestep / (30 * 24 * 60)))
        current_state = next_state

#PRINT TEST RESULTS
print('\n')
print(f'Total Energy Spent with an AI: {env.total_energy_ai}')
print(f'Total Energy Spent with an no AI: {env.total_energy_noai}')
print(f'Energy Saved: {(1 - env.total_energy_ai/env.total_energy_noai)*100}%')
print(1)