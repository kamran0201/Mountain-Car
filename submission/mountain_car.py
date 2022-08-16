'''
    1. Don't delete anything which is already there in code.
    2. you can create your helper functions to solve the task and call them.
    3. Don't change the name of already existing functions.
    4. Don't change the argument of any function.
    5. Don't import any other python modules.
    6. Find in-line function comments.

'''

import gym
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt


class sarsaAgent():
    '''
    - constructor: graded
    - Don't change the argument of constructor.
    - You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
    - Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
    - You can add more instance variable if you feel like.
    - upper bound and lower bound are for the state (position, velocity).
    - Don't change the number of training and testing episodes.
    '''

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.epsilon_T1 = 0.01 # 0.01
        self.epsilon_T2 = 0.01 # 0.01
        self.learning_rate_T1 = 0.15 # 0.15
        self.learning_rate_T2 = 0.02 # 0.02

        self.noofxT1 = 37 # 37
        self.noofvT1 = 29 # 29
        self.noofxT2 = 19 # 19
        self.noofvT2 = 15 # 15
        self.tiles = 10

        self.weights_T1 = np.zeros((self.noofxT1,self.noofvT1,3,3)) # [x,v,b]
        self.weights_T2 = np.zeros((self.tiles,self.noofxT2,self.noofvT2,3,3)) # [x,v,b]
        self.discount = 1.0
        self.train_num_episodes = 10000
        self.test_num_episodes = 100
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]

        self.xT1diff = (self.upper_bounds[0]-self.lower_bounds[0])/(self.noofxT1-1)
        self.vT1diff = (self.upper_bounds[1]-self.lower_bounds[1])/(self.noofvT1-1)
        self.xT1 = [self.lower_bounds[0]+i*self.xT1diff for i in range(self.noofxT1)]
        self.vT1 = [self.lower_bounds[1]+i*self.vT1diff for i in range(self.noofvT1)]

        self.xT2diff = (self.upper_bounds[0]-self.lower_bounds[0])/(self.noofxT2-1)
        self.vT2diff = (self.upper_bounds[1]-self.lower_bounds[1])/(self.noofvT2-1)
        self.xT2tilediff = self.xT2diff/self.tiles
        self.vT2tilediff = self.vT2diff/self.tiles
        self.xT2 = [[self.lower_bounds[0]+i*self.xT2tilediff+j*self.xT2diff for j in range(self.noofxT2)] for i in range(self.tiles)]
        self.vT2 = [[self.lower_bounds[1]+i*self.vT2tilediff+j*self.vT2diff for j in range(self.noofvT2)] for i in range(self.tiles)]

    '''
    - get_table_features: Graded
    - Use this function to solve the Task-1
    - It should return representation of state.
    '''

    def get_table_features(self, obs):
        obsx = obs[0]
        obsv = obs[1]
        closestx = min(self.xT1, key=lambda x:abs(x-obsx))
        closestv = min(self.vT1, key=lambda v:abs(v-obsv))
        return np.array([closestx,closestv,1])

    '''
    - get_better_features: Graded
    - Use this function to solve the Task-2
    - It should return representation of state.
    '''

    def get_better_features(self, obs):
        features = []
        obsx = obs[0]
        obsv = obs[1]
        for i in range(self.tiles):
            closestx = min(self.xT2[i], key=lambda x:abs(x-obsx))
            closestv = min(self.vT2[i], key=lambda v:abs(v-obsv))
            features.append([closestx,closestv,1])
        return np.array(features)

    '''
    - choose_action: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function should return a valid action.
    - state representation, weights, epsilon are set according to the task. you need not worry about that.
    '''

    def choose_action(self, state, weights, epsilon):
        action = 0
        if np.random.uniform(0, 1) < epsilon:
            action = self.env.action_space.sample()
        else:
            if weights.shape == (self.noofxT1,self.noofvT1,3,3):
                Qxv = weights[self.xT1.index(state[0])][self.vT1.index(state[1])]
                Qs0 = Qxv[0][0]*state[0]+Qxv[0][1]*state[1]+Qxv[0][2]*state[2]
                Qs1 = Qxv[1][0]*state[0]+Qxv[1][1]*state[1]+Qxv[1][2]*state[2]
                Qs2 = Qxv[2][0]*state[0]+Qxv[2][1]*state[1]+Qxv[2][2]*state[2]
                Qsa = np.array([Qs0,Qs1,Qs2])
                action = np.argmax(Qsa)
            else:
                Qs0 = 0
                Qs1 = 0
                Qs2 = 0
                for i in range(self.tiles):
                    Qxv = weights[i][self.xT2[i].index(state[i][0])][self.vT2[i].index(state[i][1])]
                    Qs0 += Qxv[0][0]*state[i][0]+Qxv[0][1]*state[i][1]+Qxv[0][2]*state[i][2]
                    Qs1 += Qxv[1][0]*state[i][0]+Qxv[1][1]*state[i][1]+Qxv[1][2]*state[i][2]
                    Qs2 += Qxv[2][0]*state[i][0]+Qxv[2][1]*state[i][1]+Qxv[2][2]*state[i][2]
                Qsa = np.array([Qs0,Qs1,Qs2])
                action = np.argmax(Qsa)
        return action

    '''
    - sarsa_update: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function will return the updated weights.
    - use sarsa(0) update as taught in class.
    - state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
    '''

    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):
        if weights.shape == (self.noofxT1,self.noofvT1,3,3):
            Qxvat = weights[self.xT1.index(state[0])][self.vT1.index(state[1])][action]
            Qstat = Qxvat[0]*state[0]+Qxvat[1]*state[1]+Qxvat[2]*state[2]
            Qxvan = weights[self.xT1.index(new_state[0])][self.vT1.index(new_state[1])][new_action]
            Qsnan = Qxvan[0]*new_state[0]+Qxvan[1]*new_state[1]+Qxvan[2]*new_state[2]
            delta = reward+self.discount*Qsnan-Qstat
            F = [state[0],state[1],state[2]]
            for i in range(3):
                Qxvat[i] += learning_rate*delta*F[i]
        else:
            Qstat = 0
            Qsnan = 0
            for i in range(self.tiles):
                Qxvat = weights[i][self.xT2[i].index(state[i][0])][self.vT2[i].index(state[i][1])][action]
                Qstat += Qxvat[0]*state[i][0]+Qxvat[1]*state[i][1]+Qxvat[2]*state[i][2]
                Qxvan = weights[i][self.xT2[i].index(new_state[i][0])][self.vT2[i].index(new_state[i][1])][new_action]
                Qsnan += Qxvan[0]*new_state[i][0]+Qxvan[1]*new_state[i][1]+Qxvan[2]*new_state[i][2]
            delta = reward+self.discount*Qsnan-Qstat
            for i in range(self.tiles):
                F = [state[i][0],state[i][1],state[i][2]]
                for j in range(3):
                    Qxvat = weights[i][self.xT2[i].index(state[i][0])][self.vT2[i].index(state[i][1])][action]
                    Qxvat[j] += learning_rate*delta*F[j]
        return weights

    '''
    - train: Ungraded.
    - Don't change anything in this function.
    
    '''

    def train(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
            weights = self.weights_T1
            epsilon = self.epsilon_T1
            learning_rate = self.learning_rate_T1
        else:
            get_features = self.get_better_features
            weights = self.weights_T2
            epsilon = self.epsilon_T2
            learning_rate = self.learning_rate_T2
        reward_list = []
        plt.clf()
        plt.cla()
        for e in range(self.train_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)
            while not done:
                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = self.choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                current_state = new_state
                if done:
                    reward_list.append(-t)
                    break
                t += 1
        self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

    '''
       - load_data: Ungraded.
       - Don't change anything in this function.
    '''

    def load_data(self, task):
        return np.load(task + '.npy')

    '''
       - save_data: Ungraded.
       - Don't change anything in this function.
    '''

    def save_data(self, task):
        if (task == 'T1'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        else:
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()

    '''
    - test: Ungraded.
    - Don't change anything in this function.
    '''

    def test(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
        else:
            get_features = self.get_better_features
        weights = self.load_data(task)
        reward_list = []
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                action = self.choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        return float(np.mean(reward_list))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
       help="first operand", choices={"T1", "T2"})
    ap.add_argument("--train", required=True,
       help="second operand", choices={"0", "1"})
    args = vars(ap.parse_args())
    task=args['task']
    train=int(args['train'])
    agent = sarsaAgent()
    agent.env.seed(0)
    np.random.seed(0)
    agent.env.action_space.seed(0)
    if(train):
        agent.train(task)
    else:
        print(agent.test(task))
