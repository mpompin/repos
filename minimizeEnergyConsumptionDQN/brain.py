from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam

class Brain(object):

    def __init__(self, learning_rate = 1e-3, number_of_actions = 5):
        self.learning_rate = learning_rate
        states = Input(shape = (3,))
        x = Dense(units = 64, activation = 'sigmoid')(states)
        x = Dropout(rate = 0.1)(x)
        y = Dense(units = 32, activation = 'sigmoid')(x)
        y = Dropout(rate=0.1)(y)
        q_values = Dense(units = number_of_actions, activation='softmax')(y)
        self.model = Model(inputs = states, outputs = q_values)
        self.model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
