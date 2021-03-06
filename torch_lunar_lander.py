from ddpg_torch import Agent

from gym.wrappers.monitoring import video_recorder
import gym
import numpy as np

RECORDING = False

env = gym.make('LunarLanderContinuous-v2')

if RECORDING:
    vid = video_recorder.VideoRecorder(env,path="./vid.mp4")


agent = Agent(alpha=0.00025, beta=0.00025, input_shape=[8], tau=0.01, env=env,
              batch_size = 64, layer1_size=400, layer2_size=300, n_actions=2)

agent.load_models()

np.random.seed(0)

score_history = []

for i in range(100):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        if RECORDING:
            env.render()
            vid.capture_frame()
        action = agent.choose_action(obs)
        new_state, reward, done, infor, = env.step(action)
        agent.remember(obs, action, reward, new_state, int(done))

        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)
    print('episode ', i, 'score %.2f' % score,
          '100 games average score %.2f' % np.mean(score_history[-100:]))

    if i % 25 == 0:
        agent.save_models()

env.close()