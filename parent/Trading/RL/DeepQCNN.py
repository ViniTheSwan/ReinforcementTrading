# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:34:21 2020

@author: vinmue
"""
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import matplotlib.animation as animation
from array2gif import write_gif
import scipy

tf.compat.v1.disable_eager_execution()


class DQNModel(object):
    def __init__(self, frame_shape, n_stacked_frames, learning_rate, n_actions, min_memory_for_training,
                 mem_size=1000000):
        self.learning_rate = learning_rate
        self.frame_shape = frame_shape
        self.n_stacked_frames = n_stacked_frames
        self.n_actions = n_actions
        self.mem_size = mem_size
        self.mem_counter = 0
        self.states = np.zeros((mem_size, *frame_shape, n_stacked_frames), dtype=np.int32)
        self.actions = np.zeros(mem_size, dtype=np.int32)
        self.rewards = np.zeros(mem_size, dtype=np.float32)
        self.new_states = np.zeros((mem_size, *frame_shape, n_stacked_frames), dtype=np.int32)
        self.dones = np.zeros(mem_size, dtype=np.int32)
        self.q_policy = None
        self.q_target = None
        self.counter = 0

    def store_transition(self, state, action, reward, new_state, done):
        if self.counter > self.n_stacked_frames:
            self.states[self.mem_counter, ...] = state
            self.actions[self.mem_counter] = action
            self.rewards[self.mem_counter] = reward
            self.new_states[self.mem_counter, ...] = new_state
            self.dones[self.mem_counter] = done
            self.mem_counter += 1
        self.counter += 1

    def sample_memory(self, batch_size):
        max_memory = min(self.mem_size, self.mem_counter)
        batch = np.random.choice(np.arange(max_memory), batch_size, replace=False)
        states = self.states[batch, ...]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        new_states = self.new_states[batch, ...]
        dones = self.dones[batch]
        return states, actions, rewards, new_states, dones

    def build_model(self, n_layers, n_neurons, output_len, learning_rate):
        self.q_policy = Sequential()
        self.q_policy.add(Conv2D(filters=8, padding='same', activation='relu', kernel_size=(3, 3),
                                 input_shape=(*self.frame_shape, self.n_stacked_frames)))
        self.q_policy.add(MaxPooling2D(pool_size=(2, 2)))
        # self.q_policy.add(Dropout(0.2))
        self.q_policy.add(Conv2D(filters=8, padding='same', activation='relu', kernel_size=(2, 2)))
        self.q_policy.add(MaxPooling2D(pool_size=(2, 2)))
        # self.q_policy.add(Dropout(0.2))
        self.q_policy.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        self.q_policy.add(Dense(256, activation='relu'))
        self.q_policy.add(Dense(256, activation='relu'))
        self.q_policy.add(Dense(self.n_actions, activation=None))  # ACTION_SPACE_SIZE = how many choices (9)
        self.q_policy.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

        self.q_target = clone_model(self.q_policy)
        tf.print(self.q_policy.summary())

    def load_model(self, model_file):
        self.q_policy = load_model(model_file)
        self.q_target = clone_model(model_file)

    def save_model(self, model_file):
        self.q_policy.save(model_file)


class DQCNNAgent(object):
    def __init__(self, learning_rate, gamma, batch_size, frame_shape, n_stacked_frames, n_actions, \
                 output_len, min_memory_for_training, epsilon, epsilon_min=0.01, epsilon_dec=1e-3, \
                 mem_size=1000000, model_file="dqn_model.h5", frozen_iterations=8):
        # input arguments
        self.it_counter = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.frame_shape = frame_shape
        self.n_stacked_frames = n_stacked_frames
        self.n_actions = n_actions
        self.output_len = output_len
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.mem_size = mem_size
        self.model_file = model_file
        self.min_memory_for_training = min_memory_for_training
        # new attributes
        self.filters = []
        self.action_space = np.arange(n_actions)
        self.dqn = DQNModel(learning_rate=learning_rate, frame_shape=frame_shape, n_stacked_frames=n_stacked_frames, \
                            n_actions=n_actions, mem_size=mem_size, min_memory_for_training=min_memory_for_training)
        self.frozen_iterations = frozen_iterations
        if os.path.exists(model_file):
            self.dqn.load_model(model_file)
        else:
            self.dqn.build_model(n_layers=3, n_neurons=128, \
                                 output_len=output_len, learning_rate=learning_rate)
        # loading model

    def save_model(self):
        self.dqn.save_model(self.model_file)

    def store_transition(self, state, action, reward, new_state, done):
        self.dqn.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        # print("states in choose action: ",state)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = state.reshape(*self.frame_shape, self.n_stacked_frames)
            state = np.expand_dims(state, axis=0)
            q_st = self.dqn.q_policy.predict(state)
            action = np.argmax(q_st)
        return action

    def save_filters(self):
        fig, ax = plt.subplots(self.filters[0].shape[-1]//2,2)
        ax = ax.ravel()
        ims = []

        for n_filt in range(self.filters[0].shape[-1]):
            ims.append(ax[n_filt].imshow(self.filters[0][0, :, :, n_filt], cmap="gray"))
        def update(frame):
            for n_filt in range(self.filters[0].shape[-1]):
                ims[n_filt].set_array((self.filters[frame][0, :, :, n_filt]))

        ani = animation.FuncAnimation(fig, update, interval=500, blit=False, frames=len(self.filters))
        ani.save(f"Introspection/filters.mp4")
        plt.close(fig)

    def filter_introspection(self, episode, buffer):
        n_frames = len(buffer)

        ### plotting filters ###
        filters, biases = self.dqn.q_policy.layers[0].get_weights()
        n_filters = filters.shape[-1]
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        self.filters.append(filters)


        ### plotting Featuremaps ###
        model1 = Sequential(self.dqn.q_policy.layers[0])
        model2 = Sequential(self.dqn.q_policy.layers[0:2])
        fig, ax = plt.subplots(3, n_filters // 2)
        ax = ax.ravel()
        ax2 = ax[-2]
        predict = model1.predict(buffer)
        ims = []
        for i in range(n_filters):
            ims.append(ax[i].imshow(predict[0, :, :, i], cmap="gray"))
        ims.append(ax2.imshow(buffer[0, :, :, 0], cmap='gray'))

        def update(frame):
            for n_filt in range(filters.shape[-1]):
                im = ims[n_filt]
                im.set_array(predict[frame, :, :, n_filt])
            ims[-1].set_array(buffer[frame, :, :, 0])
            return ims

        ani = animation.FuncAnimation(fig, update, interval=500, blit=False, frames=n_frames)
        ani.save(f"Introspection/feature_map_first_layer_episode{episode}.mp4")
        predict = model2.predict(buffer)
        ani = animation.FuncAnimation(fig, update, interval=500, blit=False, frames=n_frames)
        ani.save(f"Introspection/feature_map_two_layers_episode{episode}.mp4")
        plt.close(fig)

    def learn(self):
        if self.dqn.mem_counter < self.min_memory_for_training:
            return

        states, actions, rewards, new_states, dones = self.dqn.sample_memory(self.batch_size)
        # print("states in learn:  ",states)
        # print("new_states: ",new_states)
        q_target = self.dqn.q_policy.predict(states)
        q_next = self.dqn.q_target.predict(new_states)
        batch_index = np.arange(self.batch_size)
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * dones
        self.dqn.q_policy.fit(states, q_target, batch_size=self.batch_size, verbose=0)
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon - self.epsilon_dec \
                                                          > self.epsilon_min else self.epsilon_min
        self.it_counter += 1
        if self.it_counter % self.frozen_iterations == 0:
            self.dqn.q_target.set_weights(self.dqn.q_policy.get_weights())
        return
