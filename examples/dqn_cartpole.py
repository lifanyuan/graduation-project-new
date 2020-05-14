import numpy as np
import gym
import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from h_generation_based_on_traffic import Traffic


def plot(history, window_length=10):
    h = history.history['episode_reward']
    h_mean = [np.mean(h[i:i + window_length]) for i in range(len(h))]
    h_mean2 = [np.mean(h[i:i + 3*window_length]) for i in range(len(h))]
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    plt.plot(h, label='reward')
    plt.plot(h_mean, label='%d moving average reward' % window_length)
    plt.plot(h_mean2, label='%d moving average reward' % (3*window_length))
    plt.xlabel('x-episodes')
    plt.ylabel('y-reward history')
    plt.legend()
    # ax.set_title('Episode_reward')
    # ax.set_xlabel('episode')
    # ax = fig.add_subplot(122)
    # ax.plot(x, l)
    # ax.set_title('Loss')
    # ax.set_xlabel('episode')
    plt.show()


def save_history(history, name):
    name = os.path.join('../history', name)
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv(name, index=False, encoding='utf-8')



# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
env = Traffic(5, 2)
np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=1)  # 记忆大小
# policy = BoltzmannQPolicy()
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=199,
               target_model_update=1e-2, policy=policy, gamma=0.1)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.20000
train_history = dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

# # After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format('MECDQN'), overwrite=True)
# dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
# Finally, evaluate our algorithm for 5 episodes.
test_history = dqn.test(env, nb_episodes=20, visualize=False)
save_history(test_history, 'MECDQNhistory.csv')
ave_test = np.mean(test_history.history['episode_reward'])
print('average training reward:', '%.3f' % ave_test)
plot(train_history, 10)