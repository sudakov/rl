import random
import gym
import numpy as np
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('CartPole-v0')
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 0.01 #1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        #self.model = Sequential()
        #self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        #self.model.add(Dense(24, activation="relu"))
        #self.model.add(Dense(self.action_space, activation="linear"))
        #self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        self.model = keras.models.load_model('./m486_499')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        #self.model.save('.')
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
dqn_solver = DQNSolver(observation_space, action_space)

for i_episode in range(200):
    observation = env.reset()
    observation = np.reshape(observation, [1, observation_space])
    for t in range(700):
        action = dqn_solver.act(observation)
        env.render()
        observation_next, reward, done, info = env.step(action)
        reward = reward if not done else -reward
        observation_next = np.reshape(observation_next, [1, observation_space])
        dqn_solver.remember(observation, action, reward, observation_next, done)
        observation = observation_next
        #dqn_solver.experience_replay()
        if done:
            print(f"Run: {i_episode}, exploration_rate: {dqn_solver.exploration_rate}, score: {t}")
            break
env.close()
