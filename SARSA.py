#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done):

        self.Q_sa[s, a] += self.learning_rate * ((r + self.gamma * self.Q_sa[s_next, a_next]) - (self.Q_sa[s, a]))

        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=0.1, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    mean_return = 0

    s = env.reset()
    a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
    
    for t in range(n_timesteps):

        # Taking the action and observing the reward and next state
        s_next, r, done = env.step(a)
        
        # Choosing the next action using the current policy
        a_next = pi.select_action(s_next, policy=policy, epsilon=epsilon, temp=temp)
        
        # Q-value update
        pi.update(s, a, r, s_next, a_next, done)
        
        # goal state
        if done:
            s = env.reset()
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
        else:
            # Transition to the next state and action
            s = s_next
            a = a_next
        
        # Evaluate
        if t % eval_interval == 0:
            mean_return = pi.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(t)
        
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution
                
    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 50000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
            
    
if __name__ == '__main__':
    test()
