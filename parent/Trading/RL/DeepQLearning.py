# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:34:21 2020

@author: vinmue
"""
import numpy as np
from tensorflow.keras.layers import Dense,LSTM,Input,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

#tf.compat.v1.disable_eager_execution()
class DQNModel(object):
    def __init__(self,input_shape, action_len, mem_size=1000000):
        self.input_shape = input_shape
        self.action_len = action_len
        self.mem_size = mem_size
        self.mem_counter = 0
        self.total_counter = 0
        self.states = np.zeros((mem_size,*input_shape),dtype = np.int32)
        self.actions = np.zeros(mem_size,dtype = np.int32)
        self.rewards = np.zeros(mem_size, dtype = float)
        self.new_states = np.zeros((mem_size,*input_shape),dtype = np.int32)
        self.dones = np.zeros(mem_size,dtype = np.int32)
    def store_transition(self,state, action, reward, new_state,done):
        self.mem_counter = 0 if self.mem_counter >= self.mem_size else self.mem_counter
        self.states[self.mem_counter,:,:] = state
        self.actions[self.mem_counter] = action
        self.rewards[self.mem_counter] = reward
        self.new_states[self.mem_counter,:,:] = new_state
        self.dones[self.mem_counter] = done
        self.mem_counter+=1
        self.total_counter+=1

    def sample_memory(self,batch_size):
        max_memory = min(self.mem_size,self.total_counter)
        batch = np.random.choice(np.arange(max_memory),batch_size,replace=False)
        states = self.states[batch,:,:]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        new_states = self.new_states[batch,:,:]
        dones = self.dones[batch]
        return states,actions,rewards,new_states, dones
    def build_model(self,n_layers,n_neurons,learning_rate):
        self.q_policy = Sequential()
        #self.q_policy.add(Input(shape = (self.input_shape)))
        for i in range(n_layers):
            self.q_policy.add(LSTM(n_neurons))
            self.q_policy.add(Dropout(0.05))
        self.q_policy.add(Dense(self.action_len))
        self.q_policy.compile(optimizer =Adam(learning_rate=learning_rate),\
        loss = 'mean_squared_error')
        self.q_target = clone_model(self.q_policy)
    def load_model(self,model_file):
        self.q_policy = load_model(model_file)
        self.q_target = clone_model(model_file)
    def save_model(self,model_file):
        self.q_policy.save(model_file)


class DQNAgent(object):
    def __init__(self,learning_rate,gamma,batch_size,input_shape, action_len,\
                 min_memory_for_training, epsilon,epsilon_min = 0.01,epsilon_dec = 1e-3,\
                 mem_size=1000000, model_file = "dqn_model.h5", frozen_iterations=1):
        #input arguments
        self.it_counter =0
        self.gamma = gamma
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.action_len  = action_len
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.mem_size = mem_size
        self.model_file = model_file
        self.min_memory_for_training = min_memory_for_training
        #new attributes
        self.action_space = np.arange(action_len)
        self.dqn = DQNModel(input_shape = input_shape, action_len=action_len,mem_size= mem_size)
        self.frozen_iterations = frozen_iterations
        if os.path.exists(model_file):
            self.dqn.load_model(model_file)
        else:
            self.dqn.build_model(n_layers = 2, n_neurons = 64, learning_rate=learning_rate)
        #loading model
    def save_model(self):
        self.dqn.save_model(self.model_file)
    def store_transition(self, state, action, reward, new_state, done):
        self.dqn.store_transition(state,action,reward,new_state,done)
    def choose_action(self,state):
        #print("states in choose action: ",state)
        if np.random.random()<self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([state])
            q_st = self.dqn.q_policy.predict(state)
            action = np.argmax(q_st)
        return action
    def learn(self):
        if self.dqn.mem_counter < self.min_memory_for_training:
            return
        states, actions, rewards, new_states, dones = self.dqn.sample_memory(self.batch_size)
        #print("states in learn:  ",states)
        #print("new_states: ",new_states)
        q_target = self.dqn.q_policy.predict(states)
        q_next= self.dqn.q_target.predict(new_states)
        batch_index = np.arange(self.batch_size)
        q_target[batch_index,actions] = rewards + self.gamma * np.max(q_next, axis = 1)*dones
        self.dqn.q_policy.fit(states, q_target, batch_size = self.batch_size, verbose =0)
        #self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon - self.epsilon_dec \
        #    > self.epsilon_min else self.epsilon_min
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon * self.epsilon_dec > self.epsilon_min else self.epsilon_min
        self.it_counter += 1
        if self.it_counter % self.frozen_iterations ==0:
            self.dqn.q_target.set_weights(self.dqn.q_policy.get_weights()) 
        return