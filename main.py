from Agent import DQNAgent
from Tutorials_Agent import DQNAgent
import gym
import tensorflow as tf

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
batch_size = 32
agent = DQNAgent()

matrix = [
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
]

env = gym.make("MountainCar-v0")
print(env)


for episode in range(600):
    obs = env.reset()
    print("Episode number : ",episode)
    for step in range(10000):
        # env.render()

        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = agent.play_one_step(env, obs, epsilon)
        if done:
            break
        if episode > 35:
            agent.training_step(batch_size)
    # if episode % 100 == 0 and episode != 0:
    #     string = "DDQN model at "+ str(episode)+".h5"
    #     model.save(string)