import numpy as np
import matplotlib.pyplot as plt
import random

# Set the parameters
N = 10000
d = 9

# Create the simulation
conversion_rate = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
X = np.zeros(shape=(N, d))
for row in np.arange(N):
    for col in np.arange(d):
        if np.random.rand() <= conversion_rate[col]:
            X[row, col] = 1

#implementing a random strategy and Thompson sampling
strategies_selected_rs = []
strategies_selected_ts = []
total_reward_rs = 0
total_reward_ts = 0
numbers_of_rewards_1 = np.zeros(shape=d)
numbers_of_rewards_0 = np.zeros(shape=d)
for n in np.arange(N):
    # Random Strategy
    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    reward_rs = X[n, strategy_rs]
    total_reward_rs += reward_rs

    #Thompson sampling
    max_random = 0
    strategy_ts = 0
    for i in np.arange(d):
        random_beta = random.betavariate(alpha=numbers_of_rewards_1[i] + 1, beta= numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            strategy_ts = i
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        numbers_of_rewards_1[strategy_ts] += 1
    else:
        numbers_of_rewards_0[strategy_ts] += 1
    strategies_selected_ts.append(strategy_ts)
    total_reward_ts += reward_ts

relative_return = (total_reward_ts/total_reward_rs - 1)*100
print(f'Total Reward RS: {total_reward_rs}')
print(f'Total Reward TS: {total_reward_ts}')
print(f'Relative Improvement: {round(relative_return,2)}%')
print(1)