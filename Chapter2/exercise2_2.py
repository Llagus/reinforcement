############################################################################
# How does the softmax action selection method using the Gibbs distribution
# fare opn the 10-armed testbed?Implement the method and run it a t several
# temperatures to produce graphs similars to those in Figure 2.1.
# To verify your code, first implement the e-greedy methods and
# reproduce some specific aspect of the resultas in Figure 2.1
############################################################################
# Done by Eudald Llagostera
############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Greedy:
    def __init__(self,n_arm, reward_func):
        self.n_arm = n_arm
        self.arm_avg = np.zeros(n_arm)
        self.arm_pick = np.ones(n_arm)
        self.reward_func = reward_func
        self.rews = []
        self.its = 0

    def initialise(self):
        self.arm_avg = np.array([reward() for reward in self.reward_func])

    def update(self, its):
        self.its += its
        for iteration in range(its):
            great_arm = self.best_arm()
            reward = self.reward_func[great_arm]()
            num = (self.arm_avg[great_arm] * self.arm_pick[great_arm] + reward)
            denom = (self.arm_pick[great_arm] + 1.0)
            self.arm_avg[great_arm] = num/denom
            self.arm_pick[great_arm] += 1
            self.rews.append(reward)

    def best_arm(self):
        return np.argmax(self.arm_avg)

    def plotter(self):
        rews = np.cumsum(self.rews).astype(float)
        for i in range(len(rews)):
            rews[i] = rews[i]/(i+1.0)
        plt.plot(range(1,len(rews)+1), rews)

    def get_accumulative_rew(self):
        rews = np.cumsum(self.rews).astype(float)
        for i in range(len(rews)):
            rews[i] = rews[i] / (i + 1.0)
        return rews

class e_Greedy:
    def __init__(self, n_arm, reward_func, eps=0.01):
        self.n_arm = n_arm
        self.arm_avg = np.zeros(n_arm)
        self.arm_pick = np.ones(n_arm)
        self.reward_func = reward_func
        self.eps = eps
        self.rews = []
        self.its = 0

    def initialise(self):
        self.arm_avg = np.array([reward() for reward in self.reward_func])

    def update(self, its):
        self.its += its
        for iteration in range(its):
            random_eps = np.random.uniform(0,1)

            if random_eps < self.eps:
                arm = np.random.choice(self.n_arm,1)[0]
            else:
                arm = self.best_arm()
            reward = self.reward_func[arm]()
            num = (self.arm_avg[arm] * self.arm_pick[arm] + reward)
            denom = (self.arm_pick[arm] + 1.0)
            self.arm_avg[arm] = num/denom
            self.arm_pick[arm] += 1
            self.rews.append(reward)

    def best_arm(self):
        return np.argmax(self.arm_avg)

    def plotter(self):
        rews = np.cumsum(self.rews).astype(float)
        for i in range(len(rews)):
            rews[i] = rews[i]/(i+1.0)
        plt.plot(range(1,len(rews)+1), rews)
        plt.legend(['Rewards Greed','Rewards e-Greed'])

    def get_accumulative_rew(self):
        rews = np.cumsum(self.rews).astype(float)
        for i in range(len(rews)):
            rews[i] = rews[i] / (i + 1.0)
        return rews

if __name__ == "__main__":
    number_bandit = 2000
    number_arms = 10
    number_plays = 1000
    eps = 0.1
    funcs = np.array([lambda: np.random.normal(np.random.normal(0,1), 1) for i in range(number_arms)])
    greed = []
    e_greed = []
    dataset = list()
    dataset_egreed = list()

    for i in range(number_bandit):
        greed.append(Greedy(number_arms, funcs))
        #greed[i].initialise()
        greed[i].update(number_plays)
        rew = greed[i].get_accumulative_rew()
        dataset.append(rew)

        e_greed.append(e_Greedy(number_arms, funcs, eps))
        #e_greed[i].initialise()
        e_greed[i].update(number_plays)
        rewe= e_greed[i].get_accumulative_rew()
        dataset_egreed.append(rewe)

    result_data = np.vstack(dataset)
    result_data_e = np.vstack(dataset_egreed)
    print(result_data.shape)
    averaged = np.average(result_data, axis = 0)
    averagede = np.average(result_data_e, axis = 0)
    print(averaged.shape)
    plt.plot(range(1,len(averaged)+1), averaged)
    plt.plot(range(1, len(averagede) + 1), averagede)
    plt.legend(['Rewards Greed','Rewards e-Greed'])
    plt.show()


