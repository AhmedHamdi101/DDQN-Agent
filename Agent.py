
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import os
from tensorflow.keras import layers


matrix = np.array([
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [6, 7, 8, 9, 10],
    [10, 9, 8, 7, 6],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [6, 7, 8, 9, 10],
    [10, 9, 8, 7, 6],
    [11, 12, 13, 14, 15],
    [15, 14, 13, 12, 11],
])
next_matrix = np.zeros((10,5))

class DDQNAgent():

    def __init__(self):
        self.replay_buffer = deque(maxlen=2000)
        self.n_outputs = 3
        self.batch_size = 32
        self.discount_factor = 0.95
        self.target_update = 50
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = keras.losses.mean_squared_error

        self.model = self.create_model(input_shape=matrix.shape)    ### [ 10, 5],
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    #     train the model
    def training_step(self, batch_size):
        experiences = self.sample_experiences(batch_size)

        states, actions, rewards, next_states, dones = experiences  ## [5]   (10,5)
        next_Q_values = self.model.predict(next_states)  #next state 10*5
        #print(np.array(next_Q_values).shape)

        ## This is DQN
        max_next_Q_values = np.max(next_Q_values, axis=1)
        # print(max_next_Q_values)

        ##  This is the DDQN
        # best_next_actions = np.argmax(next_Q_values, axis=1)
        # next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()
        # next_best_Q_values = (self.target_model.predict(next_states) * next_mask).sum(axis=1)

        # target_Q_values = (np.reshape(rewards,(self.batch_size,1)) +
        #                    np.reshape((1-dones),(self.batch_size,1)) * self.discount_factor * max_next_Q_values)
        target_Q_values = (rewards +
                           (1-dones) * self.discount_factor * max_next_Q_values)

        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    #     Agent take one step in the enviroment
    def play_one_step(self, env, state, epsilon):
        state = matrix
        action = self.epsilon_greedy_policy(state, epsilon)
        # print("action is",action)
        next_state, reward, done, info = env.step(action)
        # next_state = np.transpose(matrix)
        self.replay_buffer.append((state, action, reward, next_matrix, done))
        return next_state, reward, done, info

    #     calculate the epsilon based on the greedy policy
    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(2)
        else:
            Q_values = self.model.predict(np.reshape(state,(1,state.shape[0],state.shape[1])))
            # print("Hola",Q_values)
            return np.argmax(Q_values[0])

    #     get random experiences from the queue
    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, reward, next_states, done = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)
        ]
        return states, actions, reward, next_states, done

    #     Create NN model and set one to model and one to target model
    def create_model(self,input_shape):

        # model = Sequential([
        #     Dense(32,activation="elu",input_shape=input_shape),
        #     Dense(32, activation="elu"),
        #     Dense(self.n_outputs)
        #
        # ])
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(input_shape[0]*input_shape[1], activation='relu'))
        model.add(Dense(self.n_outputs))
        # model.compile(optimizer='adam', loss='mse')


        # K = keras.backend
        # input_states = keras.layers.Input(shape=input_shape)
        # hidden1 = keras.layers.Dense(32, activation="elu")(input_states)
        # hidden2 = keras.layers.Dense(32, activation="elu")(hidden1)
        # state_values = keras.layers.Dense(1)(hidden2)
        # raw_advantages = keras.layers.Dense(self.n_outputs)(hidden2)
        # advantages = raw_advantages - K.max(raw_advantages, axis=1, keepdims=True)
        # Q_values = state_values + advantages
        # model = keras.Model(inputs=[input_states], outputs=[Q_values])
        return model
