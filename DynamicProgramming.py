#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        return argmax(self.Q_sa[s])
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''

        new_q_value = 0
        for s_next in range(self.n_states):
            new_q_value += p_sas[s_next] * (r_sas[s_next] + self.gamma * np.max(self.Q_sa[s_next]))
        self.Q_sa[s, a] = new_q_value

    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
    max_error = 0 
    delta = 1.0
    while delta > threshold:
        delta = 0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                x = QIagent.Q_sa[s, a]
                p_sas, r_sas = env.model(s, a)
                QIagent.update(s, a, p_sas, r_sas)
                delta = max(delta, abs(x - QIagent.Q_sa[s, a]))

        max_error += 1

        print("Maximum absolute error after full sweep {}: {}".format(max_error, delta))

  
    # Define the state s = 3
    s = 3

    # The optimal value for state s = 3
    ov = np.max(QIagent.Q_sa[s])

    # Printing the optimal value for state s = 3
    print("Optimal value at state s = 3:", ov)

    return QIagent

def experiment():
    gamma = 1
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    done = False
    total_reward = 0
    total_timesteps = 0
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next
        total_reward += r
        total_timesteps += 1

    mean_reward_per_timestep = total_reward / total_timesteps
    print("Mean reward per timestep under the optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()
