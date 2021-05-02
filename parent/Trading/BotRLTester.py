#from parent.Trading.RL.DeepQLearning import DQNAgent
from BotPPO import PPO
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from BotChart import BotChart
class Env:
    def __init__(self, df,  n_stacked_frames,  fees = 0.1):
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
        self.df = BotChart.get_instance().history
        self.n_stacked_frames = n_stacked_frames
        self.fees = fees
        self.prev_action = 0
        self.current_index = n_stacked_frames
        self.pairs = list(set((string.split("_")[0] for string in self.df.columns if string.split("_")[0])))
        self.render_active = False
        self.data = []
        self.balance = 1
        self.props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.colors = []
        self.palette = ['red','blue', 'green', 'black','orange', 'yellow', 'gold', 'silver', 'rose', 'violet', 'limegreen']

    def step(self, action):
        '''
        column = self.pairs[self.prev_action] + "_close"
        #print(self.df.index)
        gain = self.df.loc[self.df.index[self.current_index], column] / \
                 self.df.loc[self.df.index[self.current_index - 1],column]
        if not np.isfinite(gain):
            gain = 1
        reward = 1-gain
        '''
        if action != self.prev_action:
            sellOptions = []
            buyOptions = []
        else:
            sellOptions = [self.pairs[self.prev_action]]
            buyOptions = [self.pairs[action]]
        oldValue, ovrValue= self.strategy.sim(sellOptions,buyOptions,self.current_index)
        self.balance = ovrValue
        reward = (ovrValue/oldValue) - 1
        new_state = self.df.loc[self.df.index[self.current_index-self.n_stacked_frames+1:self.current_index+1], :].to_numpy()
        self.current_index += 1
        self.prev_action = action
        self.strategy.trade.giveInfo()
        return new_state, reward, 0, "no information"

    def render(self):
        if not self.render_active:
            plt.ion()
            self.fig = plt.figure()
            self.ax = plt.gca()
            self.render_active = True
            #print("render active")
        self.ax.clear()
        #print("new render")
        #print(self.df.index[self.current_index])
        #print(self.pairs[self.prev_action]+"_close" )
        #print(self.df.loc[self.df.index[self.current_index], self.pairs[self.prev_action]+"_close"])
        self.data.append(self.df.loc[self.df.index[self.current_index], self.pairs[self.prev_action]+"_close"])
        self.colors.append(self.palette[self.prev_action])
        self.ax.scatter(np.arange(len(self.data)), self.data, c = self.colors)
        self.ax.plot(self.data)
        self.ax.text(0.05, 1.1, f"balance: {self.balance}" \
                     , transform=self.ax.transAxes, fontsize=14, \
                     verticalalignment='top', bbox=self.props)
        plt.pause(0.7)

class BotRLTester:
    def __init__(self,):
        self.chart = BotChart.get_instance()



    #df = pd.read_csv("data/historicalData300All.csv")
    #df.columns[0] = "timestamp"
    #print(df)
    def train(self):
        #agentDQN = DQNAgent(learning_rate = 0.003 ,gamma = 0.95,batch_size = 20 ,input_shape = (N_STACKED_FRAMES,2), action_len = N_PAIRS,\
        #                 min_memory_for_training = 50000, epsilon = 1,epsilon_min = 0.01,epsilon_dec = 1e-3,\
        #                 mem_size=1000000, model_file = "dqn_model.h5", frozen_iterations=5)
        print(self.chart.history)

        env = Env( n_stacked_frames=3)

        for i in range(1000):
            if i < 20:
                new_state , reward , done , _ = env.step(0)
            else:
                new_state, reward, done, _ = env.step(1)
            env.render()
            #action = agentDQN.choose_action()
            #agentDQN.store_transition()

    def predict(self):
        pass


