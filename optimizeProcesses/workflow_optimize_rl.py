import numpy as np

# define parameters
alpha = 0.9
gamma = 0.75

# PART 1 - DEFINING THE ENVIRONMENT
# Defining the states
location_to_state = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11
}

# Defining the actions
actions = np.arange(len(location_to_state))

# Defining the rewards
R = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
]
)

# PART 2 - BUILDING THE AI SOLUTION WITH Q-LEARNING
# Initializing the Q-Values
Q = np.array(np.zeros(shape=(len(actions), len(actions))))

# Implementing the Q-Learning process
for i in np.arange(1000):
    current_state = np.random.randint(low=min(list(location_to_state.values())), high=max(list(location_to_state.values()))+1)
    playable_actions = []
    for j in actions:
        if R[current_state, j] > 0:
            playable_actions.append(j)
    next_state = np.random.choice(playable_actions)
    TD = R[current_state, next_state] + gamma*(Q[next_state, np.argmax(Q[next_state,])]) - Q[current_state, next_state]
    Q[current_state, next_state] = Q[current_state, next_state] + alpha*TD
print(f'Q-values:\n{Q.astype(int)}')

# PART 3 - GOING INTO PRODUCTION
# Making a mapping from the states to the locations
state_to_location = {state: location for location, state in location_to_state.items()}

# Making the final function that will return the optimal route
def route(starting_location, ending_location):
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state, ])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

def route_changing_R(starting_location, ending_location):
    R_new = R.copy()
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    Q = np.zeros(shape=(R_new.shape[0], R_new.shape[1]))
    for _ in np.arange(1000):
        current_state = np.random.randint(low=min(list(location_to_state.values())), high=max(list(location_to_state.values()))+1)
        playable_actions = []
        for j in actions:
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma*(Q[next_state, np.argmax(Q[next_state,])]) - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha*TD
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[next_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
    return route

# # Printing the final route
# starting_location = 'E'
# ending_location = 'G'
# print(f'Route:\n{starting_location}-->{ending_location}\n{route_changing_R(starting_location,ending_location)}')

# Making the final function that returns the optimal route
def best_route(starting_location, intermediary_location, ending_location):
    return route_changing_R(starting_location, intermediary_location) \
           + route_changing_R(intermediary_location, ending_location)[1:]

starting_location = 'E'
intermediary_location = 'K'
ending_location = 'G'
print(f'Route:\n{starting_location}-->{ending_location}\n'
      f'{best_route(starting_location, intermediary_location ,ending_location)}')
print(1)