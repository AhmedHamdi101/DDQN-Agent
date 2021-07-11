import time
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras.applications.xception import Xception


from ModifiedTensorFunction import ModifiedTensorBoard

REPLAY_MEMORY_SIZE = 2000  ## 5000  Define the size of queue whic will hold the steps
MIN_REPLAY_MEMORY_SIZE = 50 ## 1000 Define when can i start pulling out random samples (training)

MINIBATCH_SIZE = 32  ## 16

PREDICTION_BATCH_SIZE = 1  ## ???
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4 ## ???

UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Medhat"

## How much of the gpu will be used (because the rtx card try to allocate all memory)
MEMORY_FRACTION = 0.8

EPISODE = 100
DISCOUNT = 0.95 ## 0.99

EPSILON_DECAY = 0.95 ## ???
MIN_EPSILON = 0.01 ## 0.001

AGGREGATE_STATS_EVERY = 10 ## ???


# IM_WIDTH = 640
# IM_HEIGHT = 480
input_shape = [2]  ## (IM_WIDTH , IM_HEIGHT , 3)
n_outputs = 3

class DQNAgent:

    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        ## so we dont save data at each episode
        # self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        # Get updated at the end of each episode
        self.target_update_counter = 0

        # he said that he will be using different threads so he can predict and train
        # self.graph = tf.get_default_graph()  #This will be replaced with @tf.function

        self.terminate = False

        #to keep track of tensorboard
        self.last_logged_episode = 0
        self.training_intialized = False

    def update_replay_memory(self,transition):
        # transition = (current_state,action,reward,new_state,done)
        self.replay_memory.append(transition)

    def play_one_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = env.step(action)
        self.update_replay_memory((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(2)
        else:
            Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values[0])

    # @tf.function
    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) #if input is image we add ( /255)
        current_qs_list = self.model.predict(current_states,PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index,(current_state,action,reward,new_state,done) in enumerate(minibatch):
            if not done :
                max_future_q = np.max(future_qs_list[index])
                new_q = (reward + DISCOUNT * max_future_q)

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        print(X)
        tf.function(self.model.fit(X,y,batch_size=TRAINING_BATCH_SIZE,verbose=0,shuffle=False) )

        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # def get_qs(self,state):
    #     return self.model.predict(np.array(state))
    def create_model(self):

        ### need input_shape -> [size of states matrix]

        ### need n_outputs -> size of action sets
        # K = tf.keras.backend
        # input_states = Input(shape=input_shape)
        # hidden1 = Dense(32, activation="elu")(input_states)
        # hidden2 = Dense(32, activation="elu")(hidden1)
        # state_values = Dense(1)(hidden2)
        # raw_advantages = Dense(n_outputs)(hidden2)
        # advantages = raw_advantages - K.max(raw_advantages, axis=1, keepdims=True)
        # Q_values = state_values + advantages
        # model = Model(inputs=[input_states], outputs=[Q_values])
        # model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])

        model = Sequential([
            Dense(MINIBATCH_SIZE,activation="elu",input_shape=input_shape),
            Dense(MINIBATCH_SIZE,activation="elu"),
            Dense(n_outputs)
        ])
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])


        return model