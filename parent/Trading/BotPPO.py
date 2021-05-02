import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import pdb
import numpy as np
import random
from time import time
from stable_baselines3.common.vec_env import DummyVecEnv


class PPO:
    def __init__(self, config, state_probe):
        self.mem = Memory(
            config.update_every,
            config.num_env,
            config.env,
            config.device,
            config.gamma,
            config.gae_lambda
        )

        self.lr = config.lr
        self.n_steps = config.n_steps
        self.lr_annealing = config.lr_annealing
        self.gae = config.gae
        self.epsilon_annealing = config.epsilon_annealing
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.entropy_beta = config.entropy_beta
        self.device = config.device
        self.mini_batch_size = config.mini_batch_size

        config.input_shape = state_probe.shape
        self.model = ActorCriticLSTM(config).to(self.device)
        self.model_old = ActorCriticLSTM(config).to(self.device)

        self.model_old.load_state_dict(self.model.state_dict())

        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
        self.config = config

    def add_to_mem(self, state, action, reward, log_prob, values, done):
        self.mem.add(state, action, reward, log_prob, values, done)

    def act(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).to(self.device)
        else:
            x = x.to(self.config.device)
        return self.model_old.act(x)

    def learn(self, num_learn, last_value, next_done, global_step):
        # Learning Rate Annealing
        frac = 1.0 - (global_step - 1.0) / self.n_steps
        lr_now = self.lr * frac
        if self.lr_annealing:
            self.optimiser.param_groups[0]['lr'] = lr_now

        # Epsilon Annealing
        epsilon_now = self.epsilon
        if self.epsilon_annealing:
            epsilon_now = self.epsilon * frac

        # Calculate advantage and discounted returns using rewards collected from environments
        # self.mem.calculate_advantage(last_value, next_done)
        self.mem.calculate_advantage_gae(last_value, next_done)

        for i in range(num_learn):
            # itterate over mini_batches
            for mini_batch_idx in self.mem.get_mini_batch_idxs(mini_batch_size=self.mini_batch_size):

                # Grab sample from memory
                prev_states, prev_actions, prev_log_probs, discounted_returns, advantage, prev_values = self.mem.sample(
                    mini_batch_idx)
                advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # find ratios
                actions, log_probs, _, entropy = self.model.act(prev_states, prev_actions)
                ratio = torch.exp(log_probs - prev_log_probs.detach())

                values = self.model_old.get_values(prev_states).reshape(-1)

                # Stats
                approx_kl = (prev_log_probs - log_probs).mean()

                # calculate surrogates
                surrogate_1 = advantages * ratio
                surrogate_2 = advantages * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

                # Calculate losses
                new_values = self.model.get_values(prev_states).view(-1)

                value_loss_unclipped = (new_values - discounted_returns) ** 2
                values_clipped = values + torch.clamp(new_values - values, -epsilon_now, epsilon_now)
                value_loss_clipped = (values_clipped - discounted_returns) ** 2
                value_loss = 0.5 * torch.mean(torch.max(value_loss_clipped, value_loss_unclipped))

                pg_loss = -torch.min(surrogate_1, surrogate_2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss + value_loss - self.entropy_beta * entropy_loss

                # calculate gradient
                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimiser.step()

                if torch.abs(approx_kl) > 0.03:
                    break

                _, new_log_probs, _, _ = self.model.act(prev_states, prev_actions)
                if (prev_log_probs - new_log_probs).mean() > 0.03:
                    self.model.load_state_dict(self.model_old.state_dict())
                    break

        self.model_old.load_state_dict(self.model.state_dict())

        return value_loss, pg_loss, approx_kl, entropy_loss, lr_now


class Config:
    def __init__(self, env_id, env_type="gym", num_envs=8):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Running experiment {} -  on device: {}".format(env_id, self.device))
        self.seed = 1
        self.num_env = num_envs

        self.env_id = env_id

        self.win_condition = None

        self.n_steps = 7000000
        self.n_episodes = 2000
        self.max_t = 100
        self.update_every = 100

        self.epsilon = 0.1
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995

        self.gamma = 0.99
        self.lr = 1e-5
        self.hidden_size = 64
        self.mini_batch_size = 256
        self.gae = True
        self.gae_lambda = 0.95
        self.lr_annealing = False
        self.epsilon_annealing = False
        self.learn_every = 4
        self.entropy_beta = 0.01


        self.save_loc = None

        # Set up logging for tensor board
        experiment_name = f"{env_id}____{int(time.time())}"

        self.init_seed()

    def init_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


class Memory:
    def __init__(self, size, num_envs, env, device, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.size = size
        self.device = device
        self.num_envs = num_envs
        self.observation_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape

        self.states = torch.zeros((size, num_envs) + env.observation_space.shape).to(device)
        self.actions = torch.zeros((size, num_envs) + env.action_space.shape).to(device)
        self.rewards = torch.zeros((size, num_envs)).to(device)
        self.log_probs = torch.zeros((size, num_envs)).to(device)
        self.values = torch.zeros((size, num_envs)).to(device)
        self.dones = torch.zeros((size, num_envs)).to(device)

        self.idx = 0
        self.discounted_returns = None

    def add(self, states, actions, rewards, log_probs, values, dones):
        if (self.idx > self.size - 1):
            raise Exception("Memory out of space")

        self.states[self.idx] = states
        self.actions[self.idx] = actions
        self.rewards[self.idx] = torch.FloatTensor(rewards.reshape(-1)).to(self.device)
        self.log_probs[self.idx] = log_probs.reshape(-1).to(self.device)
        self.values[self.idx] = values.reshape(-1).to(self.device)
        self.dones[self.idx] = torch.FloatTensor(dones.reshape(-1)).to(self.device)

        self.idx += 1

    def calculate_discounted_returns(self, last_value, next_done):
        with torch.no_grad():
            # Create empty discounted returns array
            self.discounted_returns = torch.zeros((self.size, self.num_envs)).to(self.device)
            for t in reversed(range(self.size)):
                # If first loop
                if t == self.size - 1:
                    next_non_terminal = 1.0 - torch.FloatTensor(next_done).reshape(-1).to(self.device)
                    next_return = last_value.reshape(-1).to(self.device)
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_return = self.discounted_returns[t + 1]
                self.discounted_returns[t] = self.rewards[t] + self.gamma * next_non_terminal * next_return

    def calculate_advantage(self, last_value, next_done):
        self.calculate_discounted_returns(last_value, next_done)
        self.advantages = self.discounted_returns - self.values

    def calculate_advantage_gae(self, last_value, next_done):
        self.advantages = torch.zeros((self.size, self.num_envs)).to(self.device)
        self.discounted_returns = torch.zeros((self.size, self.num_envs)).to(self.device)

        with torch.no_grad():
            prev_gae_advantage = 0
            for t in reversed(range(self.size)):
                if t == self.size - 1:
                    next_non_terminal = 1.0 - torch.FloatTensor(next_done).reshape(-1).to(self.device)
                    next_value = last_value.reshape(-1).to(self.device)
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_value = self.values[t + 1]

                delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
                self.advantages[
                    t] = prev_gae_advantage = self.gamma * self.gae_lambda * prev_gae_advantage * next_non_terminal + delta

        self.discounted_returns = self.advantages + self.values

    def sample(self, mini_batch_idx):
        if self.discounted_returns is None or self.advantages is None:
            raise Exception("Calculate returns and advantages before sampling")

        # flatten into one array
        discounted_returns = self.discounted_returns.reshape(-1)
        states = self.states.reshape((-1,) + self.observation_shape)
        actions = self.actions.reshape((-1,) + self.action_shape)
        log_probs = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        values = self.values.reshape(-1)

        # return samples
        return states[mini_batch_idx], actions[mini_batch_idx], log_probs[mini_batch_idx], discounted_returns[
            mini_batch_idx], advantages[mini_batch_idx], values[mini_batch_idx]

    def isFull(self):
        return self.idx == self.size

    def reset(self):
        self.idx = 0

        self.states = torch.zeros((self.size, self.num_envs) + self.observation_shape).to(self.device)
        self.actions = torch.zeros((self.size, self.num_envs) + self.action_shape).to(self.device)
        self.rewards = torch.zeros((self.size, self.num_envs)).to(self.device)
        self.log_probs = torch.zeros((self.size, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.size, self.num_envs)).to(self.device)

    def get_mini_batch_idxs(self, mini_batch_size):
        # create array the size of all our data set and shuffle so indexs are at random positions
        idxs = np.arange(self.size * self.num_envs)
        np.random.shuffle(idxs)

        # create minibatches out of them of "mini_batch" size
        return [idxs[start:start + mini_batch_size] for start in
                np.arange(0, self.size * self.num_envs, mini_batch_size)]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)


class ActorCriticLSTM(nn.Module):
    def __init__(self, config):
        super(ActorCriticLSTM, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.LSTM(input_size)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, config.hidden_size)),
            nn.ReLU()
        )
        self.actor = layer_init(nn.Linear(config.hidden_size, config.action_space), std=0.01)
        self.critic = layer_init(nn.Linear(config.hidden_size, 1), std=1)

    def forward(self, x):
        return self.network(x)

    def act(self, x, action=None):
        pdb.set_trace()
        values = self.critic(self.forward(x))
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), values, probs.entropy()

    def get_values(self, x):
        return self.critic(self.forward(x))

