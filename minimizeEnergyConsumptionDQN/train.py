import os
import numpy as np
import random as rn
import environment
import brain
import dqn

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#SETTING THE PARAMETERS
epsilon = 0.3
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

env = environment.Environment(optimal_temperature= (18.0, 24.0), initial_month =0, initial_number_users = 20, initial_rate_data = 30)
brain = brain.Brain(learning_rate = 1e-5, number_of_actions = number_actions)
dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)

#choose the mode
train = True

#train AI
env.train = train
model = brain.model
if env.train:
    for epoch in np.arange(number_epochs): #epoch = 5 months
        total_reward = 0.
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _ , _ = env.observe()
        timestep = 0 #minute
        while ((not game_over) and (timestep < 5 * 30 * 24 * 60)): #exploration --> actions randomly, exploitation --> actions predicted by brain
            timestep += 1
            #PLAY NEXT ACTION BY EXPLORATION
            if np.random.rand() < epsilon:
                action = np.random.randint(0,number_actions)
                if action - direction_boundary < 0:
                    direction = -1
                else:
                    direction = 1
                energy_ai = np.abs(action - direction_boundary) * temperature_step

            # PLAY NEXT ACTION BY INFERENCEE
            else:
                q_values = model.predict(current_state.reshape(1,-1))
                action = np.argmax(q_values[0])
                if action - direction_boundary < 0:
                    direction = -1
                else:
                    direction = 1
                energy_ai = np.abs(action - direction_boundary) * temperature_step

            # UPDATE ENVIRONMENT AND REACH NEXT STATE
            next_state, reward, game_over = env.update_env(direction = direction, energy_ai = energy_ai, month = timestep//(30*24*60))
            total_reward += reward

            #STORE THIS NEW TRANSITION INTO THE MEMORY
            dqn.remember(transition=[current_state, action, reward, next_state], game_over=game_over)

            #GATHER IN TWO SEPARATE BATCHES THE INPUTS AND TARGETS
            inputs, targets = dqn.get_batch(model=model, batch_size=batch_size)

            #COMPUTE THE LOSS OVER THE TWO WHOLE BATCHES OF INPUTS AND TARGETS
            loss += model.train_on_batch(x=inputs, y=targets,)
            current_state = next_state

        #PRINT TRAIN RESULTS
        print('\n')
        print(f'Epoch:{epoch}/{number_epochs}')
        print(f'Total Energy Spent with an AI: {env.total_energy_ai}')
        print(f'Total Energy Spent with an no AI: {env.total_energy_noai}')

        #SAVE MODEL
        model.save(f'model{epoch}.h5')
