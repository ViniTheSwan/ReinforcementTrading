from parent.Trading.RL.DeepQLearning import DQNAgent
class BotRL():
    def __init__(self, nCoins):
        self.nCoins = nCoins
        self.agentDQN =  DQNAgent(learning_rate = 0.003 ,gamma = 0.95,batch_size = 20 ,state_len = nCoins, action_len = 2,\
                 min_memory_for_training = 50000, epsilon = 1,epsilon_min = 0.01,epsilon_dec = 1e-3,\
                 mem_size=1000000, model_file = "dqn_model.h5", frozen_iterations=5)

