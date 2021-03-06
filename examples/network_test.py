import numpy as np
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from h_generation_based_on_traffic import Traffic


# Get the environment and extract the number of actions.
env = Traffic(4, 4)
# np.random.seed(123)
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
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=500,
               target_model_update=1e-2, policy=policy, gamma=0.8)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# # After training is done, we save the final weights.
dqn.load_weights('../network_parameter/dqn_MECDQN_weights.h5f')

# Finally, evaluate our algorithm for some episodes.
test_history = dqn.test(env, nb_episodes=50, visualize=False)

ave_test = np.mean(test_history.history['episode_reward'])
ave_comrate = np.mean(test_history.history['computation_rate'])
print('average training reward:', '%.3f' % ave_test)
print('average computation rate:', '%.3e' % ave_comrate)

comrate_array = np.array(test_history.history['computation_rate'])
steps_array = np.array(test_history.history['nb_steps'])
comrate_per_step = comrate_array / steps_array

ave_cps = np.mean(comrate_per_step)
ave_step = np.mean(steps_array)
ave_v2i_rate = np.mean(test_history.history['v2i_rate'])
ave_conflict = np.mean(test_history.history['conflict_rate'])
ave_idle = np.mean(test_history.history['idle_rate'])
ave_abandon = np.mean(test_history.history['abandon_rate'])
ave_range = np.mean(test_history.history['out_of_range'])
print('average computation rate per step:', '%.3e' % ave_cps)
print('average steps:', '%.1f' % ave_step)
print('average v2i choosing rate:', '%.3f' % ave_v2i_rate)
print('average conflict rate:', '%.3f' % ave_conflict)
print('average idle rate:', '%.3f' % ave_idle)
print('average abandon rate:', '%.3f' % ave_abandon)
print('average range:', '%.1f' % ave_range)
