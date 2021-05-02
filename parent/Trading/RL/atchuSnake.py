# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:02:47 2020

@author: vinmue
"""
#from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
#import time
import tensorflow as tf
import os

####### Reinforcement learning ##########
#from DeepQLearning import DQNAgent
#from PolicyGradients import PolicyGradientAgent
#from DeepQLearningEager import DQAgent
#from DeepQCNN import DQCNNAgent
#from array2gif import write_gif
import atchuPolicyGradients
import matplotlib.animation as animation



##zero = empty, 1 = snake, 2 = apple
##


"global parameters"
up = 0
down = 1
left = 2
right = 3
up_vect = np.array([-1, 0])
down_vect = np.array([1, 0])
left_vect = np.array([0, -1])
right_vect = np.array([0, 1])

direction_to_string = {0: "up", 1: "down", 2: "left", 3: "right"}
direction_to_one_hot = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}

sleep = 0.025

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray




class Snake(object):
    def __init__(self, board_len):
        self.apple_eaten_this_round = False
        self.alive = True
        self.board_len = board_len
        self.direction = down
        self.direction_vect = None
        self.direction_to_vect = {up: up_vect, down: down_vect, left: left_vect, right: right_vect}
        self.len = 1
        self.body = [np.random.randint(0, board_len, 2)]
        self.ate_apple_this_round = False
    def move(self, direction):
        move = self.direction_to_vect[direction]
        self.body.append(self.body[-1] + move)
        if len(self.body) > self.len:
            self.body.pop(0)
        self.direction = direction
        self.direction_vect = move
    def eat_apple(self):
        self.len += 1
    def get_body(self):
        return self.body
    def is_alive(self):
        self.alive = True
        if len(self.body) > 1:
            if np.any(np.all(self.body[-1] == np.array(self.body[:-1]), axis=1)):
                self.alive = False
        if np.any((np.array(self.body) < 0) | (np.array(self.body) >= self.board_len), axis=None):
            self.alive = False
        return self.alive


class Statespace_manager(object):
    # self.board
    # self.apple_coords
    # self.snake
    def apple_direction(self):
        vect = self.snake.body[-1] - self.apple_coords
        max_abs = np.argmax(np.abs(vect))
        new = np.zeros(2, dtype=int)
        new[max_abs] = -np.sign(vect[max_abs])
        direction_int = self.direction_encoder[tuple(new)]
        return direction_int
    def apple_direction_one_hot(self):
        return direction_to_one_hot[self.apple_direction()]
    def snake_direction(self):
        return self.snake.direction
    def snake_direction_one_hot(self):
        return direction_to_one_hot[self.snake.direction]
    def obstacle_up(self):
        up = (np.any(np.all(self.snake.body[-1] + up_vect == np.array(self.snake.body[:]), axis=1)) \
              or np.any(0 > (self.snake.body[-1] + up_vect)) or np.any(
                    (self.snake.body[-1] + up_vect >= self.board_len)))
        return int(up)
    def obstacle_down(self):
        down = (np.any(np.all(self.snake.body[-1] + down_vect == np.array(self.snake.body[:]), axis=1)) \
                or np.any(0 > (self.snake.body[-1] + down_vect)) or np.any(
                    (self.snake.body[-1] + down_vect >= self.board_len)))
        return int(down)
    def obstacle_left(self):
        left = (np.any(np.all(self.snake.body[-1] + left_vect == np.array(self.snake.body[:]), axis=1)) \
                or np.any(0 > (self.snake.body[-1] + left_vect)) or np.any(
                    (self.snake.body[-1] + left_vect >= self.board_len)))
        return int(left)
    def obstacle_right(self):
        right = (np.any(np.all(self.snake.body[-1] + right_vect == np.array(self.snake.body[:]), axis=1)) \
                 or np.any(0 > (self.snake.body[-1] + right_vect)) or np.any(
                    (self.snake.body[-1] + right_vect >= self.board_len)))
        return int(right)
    def get_state(self):
        if self.pixels == True:
            if self.rgb_to_gray:

                self.stacked_frames.append(rgb2gray(self.board))
                if len(self.stacked_frames) > self.n_stacked_frames:
                    self.stacked_frames.pop(0)
                return np.array(self.stacked_frames)
            else:
                self.stacked_frames.append(self.board)
                if len(self.stacked_frames) > self.n_stacked_frames:
                    self.stacked_frames.pop(0)
                return np.array(self.stacked_frames)
        else:
            state = np.array(
                [*self.apple_direction_one_hot(), *self.snake_direction_one_hot(), self.obstacle_up(), self.obstacle_down(),
                self.obstacle_left(), self.obstacle_right()])
        return state
    def set_state_space_len(self):
        self.state_space_len = len(self.get_state())
    def get_state_space_len(self):
        return self.state_space_len


class Actionspace_manager(object):
    def action(self):
        return self.snake.direction

    def get_action_space_len(self):
        return self.action_space_len


class Reward_manager(object):
    def efficiency_reward(self):
        return self.apple_bonus

    def apple_reward(self):
        if self.snake.apple_eaten_this_round:
            # print("apple reward:",10)
            return 100
        else:
            return 0

    def death_reward(self):
        if not self.snake.is_alive():
            # print("death reward:", -10)
            return -10
        else:

            return 0

    def direction_reward(self):
        vect = self.apple_coords - self.snake.body[-1]
        target_dir = np.argmax(np.abs(vect))
        new_vect = np.zeros(2, dtype=int)
        new_vect[target_dir] = -np.sign(vect[target_dir])
        if np.sum(np.abs((self.snake.body[-1] - self.snake.direction_vect) - self.apple_coords)) \
                > np.sum(np.abs(self.snake.body[-1] - self.apple_coords)) or self.snake.apple_eaten_this_round:
            # print("direction_reward: ",1)
            return 1
        else:
            # print("direction_reward: ",-1)
            return -1

    def step_reward(self):
        return -0.1

    def get_reward(self):
        return self.apple_reward() + self.death_reward() + self.direction_reward()

class VideoRecorder(object):
    def __init__(self):
        self.video_frames = []
        self.scores = []

    def store(self, arr, score):
        self.video_frames.append(arr)
        self.scores.append(score)

    def save_video(self, dir_name, game_counter, highscore):
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(self.video_frames[0], animated=True)
        n_frames = len(self.video_frames)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        text = ax.text(0.05, 1.1, f"score: {self.scores[0]}, games:{game_counter}, highscore:{highscore}" \
                       , transform=ax.transAxes, fontsize=14, \
                       verticalalignment='top', bbox=props)

        def update(counter):
            text.set_text(f"score: {self.scores[counter]}, games:{game_counter}, highscore:{highscore}")
            img.set_array(self.video_frames[counter])
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=500)
        ani.save(f"Gifs/{dir_name}/episode_{game_counter}.mp4")
        plt.close(fig)

    def reset(self):
        self.video_frames = []
        self.scores = []

class Board(Statespace_manager, Reward_manager, Actionspace_manager,):
    def reset_board(self):
        self.snake = Snake(self.board_len)
        self.apple_coords = np.random.randint(0, self.board_len, 2)
        self.score = 0

    def __init__(self, board_len,dir_name,pixels=True,rgb_to_gray = False,n_stacked_frames = 1, skip=1000):
        self.video_recorder = VideoRecorder()
        if not os.path.exists(os.path.join("Gifs",dir_name)):
            os.makedirs(os.path.join("Gifs",dir_name))
        if not os.path.exists("Scores"):
            os.makedirs("Scores")
        if not os.path.exists("Models"):
            os.makedirs("Models")
        self.scores = []
        self.dir_name = dir_name
        self.n_stacked_frames = n_stacked_frames
        self.stacked_frames = []
        self.rgb_to_gray = rgb_to_gray
        self.pixels = pixels
        self.highscore = 0
        self.statespace_manager = Statespace_manager()
        self.reward_manager = Reward_manager()
        self.apple_bonus = board_len
        self.game_counter = 0
        self.action_space = np.array([up, down, left, right])
        self.action_space_len = len(self.action_space)
        self.state_space_len = 0
        ####### for skipping transitions
        self.skip = skip
        self.step_counter = 0
        ###########################
        self.direction_encoder = {tuple(up_vect): up, tuple(down_vect): down, tuple(left_vect): left,
                                  tuple(right_vect): right}
        # plt.ion()
        self.score = 0
        self.fig = plt.gcf()
        self.ax = plt.gca()
       
        #self.camera = Camera(self.fig)
        # self.fig.canvas.mpl_connect('close_event', self.closeHandler)
        self.imggg = None
        self.props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.ax.text(0.05, 0.95, "score:", transform=self.ax.transAxes, fontsize=14, \
                     verticalalignment='top', bbox=self.props)
        self.board_len = board_len
        self.board = np.zeros((board_len, board_len, 3))
        self.snake = Snake(board_len)
        self.apple_coords = np.random.randint(0, board_len, 2)
        self.img = plt.imshow(self.board, interpolation='none', \
                              extent=[0, len(self.board), 0, len(self.board)])
        self.ate_apple_this_round = False
        ###################ugly
        self.set_state_space_len()
        ###################

    def get_board_len(self):
        return self.board_len

    def get_apple_coords(self):
        return self.apple_coords

    def get_snake(self):
        return self.snake

    def set_apple(self):
        while np.any(np.all(self.apple_coords == np.array(self.snake.get_body()), axis=1)):
            self.apple_coords = np.random.randint(0, self.board_len, 2)

    def print_board(self):
        if self.step_counter == self.skip:
            plt.ion()
        if self.step_counter >= self.skip:
            self.ax.clear()
            self.ax.text(0.05, 1.1, f"score: {self.score}, games: {self.game_counter}, highscore: {self.highscore}" \
                         , transform=self.ax.transAxes, fontsize=14, \
                         verticalalignment='top', bbox=self.props)
            self.ax.imshow(self.board, interpolation='none')
            plt.pause(0.025)

    def update_board(self):
        self.board[:, :, :] = np.array([1., 1., 1.])  # resetting board
        self.board[self.apple_coords[0], self.apple_coords[1]] = np.array([1., 0, 0])  # printing apple
        for i in range(len(self.snake.body[:-1])):
            loc = self.snake.body[i]
            self.board[loc[0], loc[1], :] = np.array([0., 1., 0.])
            # i/(2*len(self.snake.get_body()))
        # print(self.board)
        self.board[self.snake.get_body()[-1][0], self.snake.get_body()[-1][1], :] = np.array([0., 0., 0.])

    def check_apple(self):
        if self.apple_bonus >= 1:
            self.apple_bonus -= 1
        if np.all(self.snake.body[-1] == self.apple_coords):
            self.snake.eat_apple()
            self.set_apple()
            self.score += 1
            self.highscore = max(self.score, self.highscore)
            self.snake.apple_eaten_this_round = True
            self.apple_bonus = self.board_len
        else:
            self.snake.apple_eaten_this_round = False

    def play(self, direction):
        ######## DONE ########
        done = 1
        ######################
        ######## STATE  #######
        state = self.get_state()
        #######################
        ######## ACTION #######
        action = direction
        #######################
        self.snake.move(direction)
        self.check_apple()
        ######## REWARD ########
        reward = self.get_reward()
        ########################
        #self.canvas.draw()
        self.video_recorder.store(np.copy(self.board), self.score)
        if not self.snake.is_alive():
            # plt.text(0,0,"game_over")
            done = 0
            self.scores.append(self.score)
            self.reset_board()
            self.set_apple()
#            self.scores.append(self.score)
            if board.game_counter % 100 == 0:
                eps = np.arange(self.game_counter - len(self.scores) + 1, self.game_counter + 1, 1)
                res = np.array([eps, self.scores], dtype=int)
                with open(f'Scores/{self.dir_name}.csv', 'ab') as f:
                    np.savetxt(f, res.T, delimiter=",", fmt='%d')
                self.scores = []
                self.video_recorder.save_video(self.dir_name,self.game_counter,self.highscore)
            self.game_counter += 1
            self.video_recorder.reset()
        self.update_board()
        self.print_board()
        self.step_counter += 1
        ######## NEW STATE ########
        new_state = self.get_state()
        ############################
        return state, action, reward, new_state, done

if __name__ == "__main__":
    print("tf is in eager mode: ",tf.executing_eagerly())

    board = Board(15, skip=10_000, pixels = False, rgb_to_gray=False, dir_name = "PolicyGradient",  n_stacked_frames = 3)

    agentPolicy = atchuPolicyGradients.model(0.99)
    

#atchus Policy Gradient Agent

    state = board.get_state()
    for s in range(20000):
        done = False
        total_reward = 0
        # three list to keep track of reward state and actions
        rewards = []
        states = []
        actions = []
        while not done:
            # env.render()
            action = agentPolicy.act(state)
            # print(action)
            state, action, reward,next_state, done= board.play(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            total_reward += reward

            if done:  # model only trains after an epsiode not in every step (don't know if correct)
                agentPolicy.train(states, rewards, actions)
                # print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {}".format(s, total_reward))


