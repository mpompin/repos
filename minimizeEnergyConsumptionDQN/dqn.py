import numpy as np

class DQN(object):

    # INTRODUCE AND INITIALIZE ALL PARAMETERS AND VARIABLES OF THE ENVIRONMENT
    def __init__(self, max_memory = 100, discount = 0.9):
        self.memory = []
        self.max_memory = max_memory
        self.discount = discount

    # METHOD THAT BUILDS MEMORY IN EXPERIENCE REPLAY
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    #METHOD THAT BUILDS TWO BATCHES OF 10 INPUTS AND 10 TARGETS BY EXTRACTING 10 TRANSITIONS
    def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0][0].shape[0]
        num_outputs = model.output_shape[-1]
        inputs = np.zeros(shape=(min(batch_size, len_memory), num_inputs))
        targets = np.zeros(shape=(min(batch_size, len_memory), num_outputs))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(batch_size, len_memory))):
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            q_next_state = np.max(model.predict(next_state)[0])
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount*q_next_state


