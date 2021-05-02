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
    def __init__(self,state_len,mem_size):
        self.state_len = state_len
        self.mem_size = mem_size
        self.mem_counter = 0
        self.states = np.zeros((mem_size,state_len),dtype = np.int32)
        self.actions = np.zeros(mem_size,dtype = np.int32)
        self.rewards = np.zeros(mem_size, dtype = float)
        self.new_states = np.zeros((mem_size,state_len),dtype = np.int32)
        self.dones = np.zeros(mem_size,dtype = np.int32)
    def store_transition(self,state, action, reward, new_state,done):
        self.states[self.mem_counter,:] = state
        self.actions[self.mem_counter] = action
        self.rewards[self.mem_counter] = reward
        self.new_states[self.mem_counter,:] = new_state
        self.dones[self.mem_counter] = done
        self.mem_counter+=1
    def sample_memory(self,batch_size):
        max_memory = min(self.mem_size,self.mem_counter)
        batch = np.random.choice(np.arange(max_memory),batch_size,replace=False)
        states = self.states[batch,:]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        new_states = self.new_states[batch,:]
        dones = self.dones[batch]
        return states,actions,rewards,new_states, dones
        
class DQNetwork(keras.Model):
    def get_config(self):
        pass
    def __init__(self,state_len, n_actions,learning_rate, **layers_layout):
        super(DQNetwork,self).__init__()
        self.Input = Input(shape = (None, state_len))
        self.layers_layout = layers_layout
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.layers_layout = layers_layout
        for name, layer in layers_layout.items():
            setattr(self, name,eval(layer))
            print(name)
        self.q = Dense(n_actions,activation =None)
    #@tf.function
    def call(self,state):
        value = state
        for name, layer in self.layers_layout.items():
            value = getattr(self,name)(value)
        q = self.q(value)
        return q
    

    

class DQAgent(object):
    def __init__(self,learning_rate,gamma,batch_size,state_len, n_actions, min_memory_for_training,
                 epsilon,epsilon_min = 0.01,epsilon_dec = 1e-3,
                 mem_size=1000000, model_file = "dqn_model.h5", frozen_iterations=8, **layers_layout):
        #input arguments
        self.it_counter =0
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
        #new attributes
        self.q_policy = DQNetwork(state_len, n_actions,learning_rate,**layers_layout)
        self.q_target = DQNetwork(state_len, n_actions,learning_rate,**layers_layout)
        self.q_target.set_weights(self.q_policy.get_weights())
        self.q_policy.compile(loss="mean_squared_error", optimizer=Adam(learning_rate))
        self.q_target.compile(loss="mean_squared_error", optimizer=Adam(learning_rate))
        self.replay_buffer = ReplayBuffer(self.state_len,mem_size)
        #########
        self.frozen_iterations = frozen_iterations
        #loading model TODO
    def store_transition(self, state, action, reward, new_state, done):
        self.replay_buffer.store_transition(state,action,reward,new_state,done)
    #@tf.function
    def choose_action(self,state):
        #print("states in choose action: ",state)
        if np.random.random()<self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            state = state.reshape(1,len(state))
            q_st = self.q_policy(state)
            action = np.argmax(q_st)
        return int(action)

    def learn(self):
        if self.replay_buffer.mem_counter < self.min_memory_for_training:
            return
        states, actions, rewards, new_states, dones = self.replay_buffer.sample_memory(self.batch_size)
        #print("states in learn:  ",states)
        #print("new_states: ",new_states)
        states = tf.convert_to_tensor(states,dtype = tf.float32)
        actions = tf.convert_to_tensor(actions,dtype = tf.int64)
        rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype = tf.float32)
        dones = tf.convert_to_tensor(dones, dtype = tf.float32)
        gamma = tf.constant(self.gamma)
        with tf.GradientTape() as tape:
            loss = 0
            for i,(state, action, reward,new_state, done) in enumerate(zip(states, actions, rewards,new_states, dones)):
                state = tf.expand_dims(state, axis =0)
                new_state = tf.expand_dims(new_state,axis =0)
                q_eval = tf.squeeze(self.q_policy(state))[action]
                q_next = tf.squeeze(tf.reduce_max(self.q_target(new_state)))
                q_target = reward + gamma * q_next
                loss += tf.pow(q_target-q_eval,2)
        gradient = tape.gradient(loss, self.q_policy.trainable_variables)
        self.q_policy.optimizer.apply_gradients(zip(gradient, self.q_policy.trainable_variables))
        self.epsilon = self.epsilon- self.epsilon_dec if self.epsilon - self.epsilon_dec \
            > self.epsilon_min else self.epsilon_min
        self.it_counter += 1
        if self.it_counter % self.frozen_iterations == 0:
            self.q_target.set_weights(self.q_policy.get_weights())
        return