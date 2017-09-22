"""
using hrl for dialogure manguagement
depends:
python2+/python3
test on ubuntu 16.04

"""
import numpy as np
import math
import gym
import os.path
import pickle as pkl
from sklearn.base import clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C     # kernal for GP
# from scipy.linalg import cholesky, cho_solve, solve_triangular
# from scipy.optimize import fmin_l_bfgs_b     # minimize a function
# import pandas as pd
from maze_env import Maze
from copy import deepcopy

np.random.seed(1)  # for reproduction

# define parameters
N_EPISODE = 100
GAMMA = .9
E_GREEDY = .1

# ====================param for dialogue===============
# n_belief_restaurant_hrl = 311  # states
# n_belief_hotel_hrl = 156
# n_belief_book_hrl = 431
# n_belief_pay_hrl = 174

# n_belief_restaurant_flat = 490
# n_belief_hotel_flat = 333
# n_action = []


# =============environment for dialogue system, not implemented===========


class Diag_env(object):
    """docstring for environment"""

    def __init__(self):
        self.action_space = []
        self.n_actions = len(self.action_space)
        self.title('***')

    def step(self, action):
        # dynamics, return the next state, reward, episode is / not
        # terminal
        pass

    def reset(self):
        # intinalize before start or dialogure is terminate
        pass

    def seed(self):
        # set random seed
        pass

    def render(self):
        pass

    def is_over(self):
        pass


# ===============================define dictionary class=================
class Memory(object):

    def __init__(self):
        self.data = []
        self.reward = []
        self.index = 0
        self.capacity = len(self.data)

    def store_transition(self, r_, s, a, save_s_a=True):
        if save_s_a:
            transition = np.hstack((s, a))
        else:
            transition = np.hstack((s, a,  r_))
        self.reward.append(r_)
        self.data.append(transition)
        self.index += 1
        self.capacity = self.index

    def samle(self, n):
        # assert self.index >= self.capacity, 'error, have empty memory'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

    def check_exist(self, s_, a_):
        if np.hstack((s_, a_)) in self.data:
            return True

    def measure(self, s, a):
        # function used for constrain the capacity of our data. As we know if store every date we have met,
        # the kernal matrix will bocome bigger and bigger. To solve this, some measure can be deduced to
        # decide if we shoud add the current sate action pair to our memory
        pass


# =====================================learning algorithm=================
class Agent(object):

    def __init__(self, actions, gp_sarsa=None, gamma=GAMMA, epsilon=E_GREEDY):
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        # self.learner = gp_sarsa

        # if self.learner == None:
        #     self.learner = GPSARSA()

    def choose_action(self, s, learner, step, episode, stragedy='e_greedy'):
        # action selection ,useing epsilon greedy policy
        if stragedy == 'e_greedy':
            if np.random.rand() > self.epsilon:
                Q = []
                for action in self.actions:
                    mean, _ = learner.Q_esitimate(s, action)
                    Q.append(mean)
                Q = np.array(Q)
                print('Q for step %d  episode %d is: %s' % (step, episode, Q))
                action = Q.argmax()  # to do ..........return index
            else:
                # choose random
                action = np.random.choice(self.actions)

        else:
            # covariance based policy
            pass
        return action

    def act(self, a):
        return self.env.step(a)

    def learn(self):
        # using gpsarsa class
        pass


# ===========================================================================
class GPSARSA(object):
    """
    docstring for GPSARSA
    class used for implement GPSARSA algorithm

    """

    def __init__(self, s, a, r, regularization=1e-5, kernal=None, gamma=GAMMA, memory=None):

        self.sigma = regularization     # noise for Q(s,a)
        self.kernal = kernal
        self.gamma = gamma
        self.B_t = memory
        self.s = s
        self.a = a
        self.r = r

        if self.kernal is None:
            self.kernal_ = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-4, 1e4))
        else:
            self, kernal_ = clone(self.kernal)

        if self.B_t is None:
            self.B_t = Memory()

    def optimaize_theta(self):
        # used for optimize paramter in kenal function
        pass

    def update_param(self, r_, s_, a_, update_states=True):
        '''
        # B_t is state-cations pairs have been visited shape=(n_samples, n_features)
        # r_t  is correspondence value shape=(n_samples, 1)
        # t is time step in an episode
        # quatities required for predictions for query point
        # update memory by inserting s_ and a_
        '''
        if update_states:
            self.B_t.store_transition(r_, s_, a_)
            B_t = np.array(self.B_t.data)
        else:
                # create a temporay instance used for estimate current action
            fake_mem = deepcopy(self.B_t)
            fake_mem.store_transition(r_, s_, a_)
            B_t = np.array(fake_mem.data)
        s_a_ = np.hstack((s_, a_))
        dic_size = B_t.shape[0]     # size of our dictionary
        k_cap = self.k_cap_       # last K
        m, n = k_cap.shape
        m_ = m + 1
        n_ = n + 1
        k_cap_t = np.zeros((m_, n_))

        # column vector, consistent with paper
        k_t = self.kernal_(s_a_, B_t).T
        k_cap_t[0:m, 0:n] = k_cap
        k_cap_t[0:m_, n:n_] = k_t
        k_cap_t[m:m_, 0:n_] = k_t.T
        # print(k_cap_t)

        # n+1 states for n-th reward
        H_t_ = np.zeros((dic_size - 1, dic_size))
        for i in range(dic_size - 1):
            H_t_[i, i] = 1
            H_t_[i, i + 1] = -self.gamma

        # update memory library

        R_ = np.array(self.B_t.reward)
        # R_ = np.array(R_.append(self.r))
        k = self.kernal_(s_a_)

        self.k_t = k_t

        self.H_t = H_t_
        self.R = R_
        self.k = k
        self.k_cap_t = k_cap_t

        if update_states:
            self.s = s_
            self.a = a_
            self.r = r_
            self.k_cap_ = k_cap_t

    def Q_esitimate(self, s_, a_, update=False):
        self.update_param(self.r, s_, a_, update_states=update)
        temp = self.H_t.dot(self.k_cap_t).dot(self.H_t.T) + \
            np.square(self.sigma) * np.dot(self.H_t, self.H_t.T)
        try:
            np.linalg.inv(temp)
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.message:
                print('matrix is sigular')

        Q_mean = self.k_t.T.dot(self.H_t.T).dot(
            np.linalg.inv(temp)).dot(self.R)
        Q_var = self.k - \
            self.k_t.T.dot(self.H_t.T).dot(
                np.linalg.inv(temp)).dot(self.H_t).dot(self.k_t)
        return Q_mean.flatten(), Q_var


def update():
    for episode in range(N_EPISODE):
        step = 1
        s = env.reset()
        if episode == 0:    # first episode
            a = np.random.choice(agent.actions)
            s_, r_, done = env.step(a)
            s_a_ = np.hstack((s, a))
            gpsarsa = GPSARSA(s, a, r_)    # create a learner instance
            gpsarsa.B_t.store_transition(r_, s, a)
            gpsarsa.k_cap_ = gpsarsa.kernal_(s_a_)
            gpsarsa.r = r_
            gpsarsa.H_t = np.array([1 - gpsarsa.gamma])
            s = s_

        while True:
            env.render()

            a_ = agent.choose_action(
                s=s, learner=gpsarsa, step=step, episode=episode)
            step += 1

            # take action and get the next observation
            obs_, reward, done = env.step(a_)
            gpsarsa.r = reward
            gpsarsa.update_param(r_, s, a_)

            s = obs_
            r_ = reward

            if done:
                break
    # end
    print('game over')
    env.destory()


if __name__ == '__main__':
    env = Maze()
    agent = Agent(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
