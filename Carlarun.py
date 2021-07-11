from Agent import DDQNAgent

agent = DDQNAgent()

for episode in range(100):
    ## you may add colision_hist = []

    obs = env.reset()
    episode_reward = 0
    step = 0
    print("Episode",episode)
    while True:

        epsilon = max(1 - episode / 500, 0.01)

        obs, reward, done, info = agent.play_one_step(env, obs, epsilon)
        episode_reward += reward
        step += 1

        if done:
            break

        if episode > 50:
            agent.training_step(batch_size=32)

        if agent.target_model_counter == 50:
            agent.target_model.set_weights(agent.model.get_weights())
            agent.target_model_counter = 0

    agent.target_model_counter += 1