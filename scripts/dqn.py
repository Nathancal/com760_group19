#!/usr/bin/env python
#################################################################################
# Code based on file provided by Robotis https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning/tree/master/turtlebot3_dqn

import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from com760_group19.msg import Group19DqnCustom, Group19DqnResultCustom # Import the custom message
from src.dqn_env import DQNEnv
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation

EPS = 5000

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.pub_action_execution = rospy.Publisher('action_execution', Group19DqnCustom, queue_size=5)  # Using custom message
        self.pub_result = rospy.Publisher('result', Group19DqnResultCustom, queue_size=5)  # Publish custom result message
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('com760_group19/scripts', 'com760_group19/save/group19_')
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.0002
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=1000000)

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()

        if self.load_model:
            try:
                self.model.set_weights(load_model(self.dirPath + str(self.load_episode) + ".h5").get_weights())

                with open(self.dirPath + str(self.load_episode) + '.json') as outfile:
                    param = json.load(outfile)
                    self.epsilon = param.get('epsilon')
            except Exception as e:
                rospy.logerr("Error loading model: %s", str(e))

    def buildModel(self):
        model = Sequential()
        dropout = 0.2

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))

        model.add(Dense(128, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(256, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))
        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model
    
    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)

if __name__ == '__main__':
    rospy.init_node('dqn_node')
    pub_result = rospy.Publisher('result', Group19DqnResultCustom, queue_size=5)
    pub_action_execution = rospy.Publisher('action_execution', Group19DqnCustom, queue_size=5)  # Using custom message
    result = Group19DqnResultCustom()

    state_size = 28
    action_size = 5

    env = DQNEnv(action_size)

    agent = DQNAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    for e in range(agent.load_episode + 1, EPS):
        done = False
        state = env.reset()
        score = 0
        try:
            for t in range(agent.episode_step):
                action = agent.getAction(state)

                next_state, reward, done = env.move_robot(action)

                agent.appendMemory(state, action, reward, next_state, done)

                if len(agent.memory) >= agent.train_start:
                    if global_step <= agent.target_update:
                        agent.trainModel()
                    else:
                        agent.trainModel(True)

                score += reward
                state = next_state
                
                # Publishing action execution
                action_execution_msg = Group19DqnCustom()
                action_execution_msg.action = action
                action_execution_msg.score = score
                action_execution_msg.reward = reward
                pub_action_execution.publish(action_execution_msg)

                if e % 10 == 0:
                    agent.model.save(agent.dirPath + str(e) + '.h5')
                    with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                        json.dump(param_dictionary, outfile)

                if t >= 400:
                    rospy.loginfo("group19Bot took too long..")
                    done = True

                if done:
                    result.score = score
                    result.max_q_value = np.max(agent.q_value)
                    pub_result.publish(result)
                    agent.updateTargetModel()
                    scores.append(score)
                    episodes.append(e)
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)

                    rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                                  e, score, len(agent.memory), agent.epsilon, h, m, s)
                    param_keys = ['epsilon']
                    param_values = [agent.epsilon]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                global_step += 1
                if global_step % agent.target_update == 0:
                    agent.updateTargetModel()
                    rospy.loginfo("UPDATE TARGET NETWORK")

            if agent.epsilon > agent.epsilon_min:

                if e % 3 == 0:
                    agent.epsilon *= agent.epsilon_decay

        except Exception as ex:
            rospy.logerr("Error in episode {}: {}".format(e, str(ex)))
            continue
