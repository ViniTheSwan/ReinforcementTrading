# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 16:33:13 2020

@author: atchu
"""

from gym.envs.registration import register

import numpy as np
import gym  # random import to model sth
import tensorflow as tf
import tensorflow_probability as tfp

#env = gym.make('snake-v0')


class model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(30, activation='relu')
        self.d2 = tf.keras.layers.Dense(30, activation='relu')
        self.out = tf.keras.layers.Dense(4,
                                         activation='softmax')  # output probabilities for each action (Lec 6 Slide 7)

    def call(self, input_data):
        x = tf.convert_to_tensor(input_data)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x

    def act(self, state):
        """
        input : numpy array of states
        output: predicted probabilities
        tfp turn probabilities into a distribution
        """
        prob = self(np.array([state]))
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()  # sampling from distribution
        return int(action.numpy()[0])  # action returned as integer
    @tf.function
    def a_loss(self, prob, action, reward):
        """
        loss = -(prob of selected action*discounted reward)
        """
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward  # isch gradient ascent wenn ich mich ned tüsche ned descent
        return loss

    def train(self, states, rewards, actions):
        """
        input: list of states, actions, rewards
        output: calculates gradient
        """
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()  # starting from the last element
        for r in rewards:  # calculates the cumulative expected reward for each state
            sum_reward = r + self.gamma * sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()

        for state, reward, action in zip(states, discnt_rewards, actions):
            with tf.GradientTape() as g:
                p = self.model(np.array([state]), training=True)
                loss = self.a_loss(p, action, reward)
            grads = g.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads,
                                         self.model.trainable_variables))  # opt.apply_gradient ish en optimizierti variante zum gradient descent berechne


# Main loop
if __name__ == "main":
    agent1 = agent()
    steps = 500  # number of steps in a episode
    for s in range(steps):
        done = False
        state = env.reset()
        total_reward = 0
        # three list to keep track of reward state and actions
        rewards = []
        states = []
        actions = []
        while not done:
            # env.render()
            action = agent1.act(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            total_reward += reward

            if done:  # model only trains after an epsiode not in every step (don't know if correct)
                agent1.train(states, rewards, actions)
                # print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {}".format(s, total_reward))