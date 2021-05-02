# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:23:25 2020

@author: vinmue
"""

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import load_model, clone_model
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import clone_model


class ReplayBuffer(object):
    def __init__(self, frame_shape, n_stacked_frames,  mem_size):
        self.frame_shape = frame_shape
        self.n_stacked_frames = n_stacked_frames
        self.mem_size = mem_size
        self.mem_counter = 0
        self.states = np.zeros((mem_size,*frame_shape, n_stacked_frames ), dtype=np.int32)
        self.actions = np.zeros(mem_size, dtype=np.int32)
        self.rewards = np.zeros(mem_size, dtype=float)
        self.new_states = np.zeros((mem_size,*frame_shape, n_stacked_frames), dtype=np.int32)
        self.dones = np.zeros(mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, new_state, done):
        self.states[self.mem_counter, ...] = state
        self.actions[self.mem_counter] = action
        self.rewards[self.mem_counter] = reward
        self.new_states[self.mem_counter, ...] = new_state
        self.dones[self.mem_counter] = done
        self.mem_counter += 1

    def sample_memory(self, batch_size):
        max_memory = min(self.mem_size, self.mem_counter)
        batch = np.random.choice(np.arange(max_memory), batch_size, replace=False)
        states = self.states[batch, :]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        new_states = self.new_states[batch, :]
        dones = self.dones[batch]
        return states, actions, rewards, new_states, dones


class DQNetwork(keras.Model):
    def get_config(self):
        pass

    def __init__(self, state_len, n_actions, learning_rate, **layers_layout):
        super(DQNetwork, self).__init__()
        self.Input = Input(shape=(None, state_len))
        self.layers_layout = layers_layout
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.layers_layout = layers_layout
        for name, layer in layers_layout.items():
            setattr(self, name, eval(layer))
            print(name)
        self.q = Dense(n_actions, activation=None)

    # @tf.function
    def call(self, state):
        value = state
        for name, layer in self.layers_layout.items():
            value = getattr(self, name)(value)
        q = self.q(value)
        return q


class DQAgent(object):
    def __init__(self, learning_rate, gamma, batch_size, state_len, n_actions, min_memory_for_training,
                 epsilon, epsilon_min=0.01, epsilon_dec=1e-3,
                 mem_size=1000000, model_file="dqn_model.h5", frozen_iterations=8, **layers_layout):
        # input arguments
        self.it_counter = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.state_len = state_len
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.mem_size = mem_size
        self.model_file = model_file
        self.min_memory_for_training = min_memory_for_training
        # new attributes
        self.q_policy = DQNetwork(state_len, n_actions, learning_rate, **layers_layout)
        self.q_target = DQNetwork(state_len, n_actions, learning_rate, **layers_layout)
        self.q_target.set_weights(self.q_policy.get_weights())
        self.q_policy.compile(loss="mean_squared_error", optimizer=Adam(learning_rate))
        self.q_target.compile(loss="mean_squared_error", optimizer=Adam(learning_rate))
        self.replay_buffer = ReplayBuffer(self.state_len, mem_size)
        #########
        self.frozen_iterations = frozen_iterations
        # loading model TODO

    def store_transition(self, state, action, reward, new_state, done):
        self.replay_buffer.store_transition(state, action, reward, new_state, done)

    #@tf.function
    def choose_action(self, state):
        # print("states in choose action: ",state)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = state.reshape(1, len(state))
            q_st = self.q_policy(state)
            action = np.argmax(q_st)
        return action

    #@tf.function
    def learn(self):
        if self.replay_buffer.mem_counter < self.min_memory_for_training:
            return
        states, actions, rewards, new_states, dones = self.replay_buffer.sample_memory(self.batch_size)
        # print("states in learn:  ",states)
        # print("new_states: ",new_states)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int64)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        # q_next = self.q_target(new_states)
        gamma = tf.constant(self.gamma)
        with tf.GradientTape() as tape:
            '''
            gamma = tf.constant(self.gamma)
            q_next = self.q_target(new_states)
            q_target = self.q_policy(states)
            q_target_updated = rewards + gamma * tf.math.multiply(tf.math.reduce_max(q_next, axis = 1),dones)
            loss = tf.losses.mean_squared_error(q_target_updated,q_target[:,actions])
            '''
            loss = 0
            for i in range(states.shape[0]):
                state = states[i, :]
                action = actions[i]
                reward = rewards[i]
                done = dones[i]
                new_state = new_states[i]

                ''''''
                state = tf.expand_dims(state, axis=0)
                q_target = self.q_policy(state)
                q_next = self.q_target(new_states)
                # q_target_updated = reward + gamma * tf.math.reduce_max(q_next) * done
                # loss +=  tf.pow((q_target_updated - tf.squeeze(q_target)[action]),2)
                ###############

                q_target_action = reward + gamma * tf.math.reduce_max(q_next) * done
                q_target_action = tf.reshape(q_target_action, (1, 1))
                q_target_updated = tf.concat([q_target[:, :action], q_target_action, q_target[:, action + 1:]], axis=1)
                loss += tf.losses.mean_squared_error(q_target_updated, q_target)
                # tf.print(loss.shape)
                # loss += (tf.math.reduce_max(q_next-q_target))**2

        gradient = tape.gradient(loss, self.q_policy.trainable_variables)
        self.q_policy.optimizer.apply_gradients(zip(gradient, self.q_policy.trainable_variables))
        '''
        q_target = self.dqn.q_policy.predict(states)
        q_next= self.dqn.q_target.predict(new_states)
        batch_index = np.arange(self.batch_size)
        q_target[batch_index,actions] = rewards + self.gamma * np.max(q_next, axis = 1)*dones
        self.dqn.q_policy.fit(states, q_target, batch_size = self.batch_size, verbose =0)
        '''
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon - self.epsilon_dec \
                                                          > self.epsilon_min else self.epsilon_min
        self.it_counter += 1
        if self.it_counter % self.frozen_iterations == 0:
            self.q_target.set_weights(self.q_policy.get_weights())
        return