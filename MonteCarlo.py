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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''

        G = 0
        
        T_ep = (len(states) - 1) 

        for t in reversed (range(T_ep)):

            G = rewards[t] + self.gamma * G 
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - self.Q_sa[states[t], actions[t]])

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []


    # Initialize total_t
    total_t = 0  

    while total_t < n_timesteps:
        s = env.reset()

        # Creating array for states, actions and rewards
        s_ep = []
        a_ep = []
        r_ep = []

        for t in range(max_episode_length): 

            # Sample action
            a = pi.select_action(s, policy, epsilon, temp)

            # Simulate environment
            s_next, r, done = env.step(a)

            # Appending states, actions and rewards in array from the episode
            s_ep.append(s)
            a_ep.append(a)
            r_ep.append(r)

            # Incrementing total_t
            total_t += 1

            # Evaluate
            if (total_t) % eval_interval == 0:

                mean_return = pi.evaluate(eval_env)
                eval_timesteps.append(total_t)
                eval_returns.append(mean_return)
        
            # Goal state
            if done: 
                break
            s = s_next

        # Update Q-value
        pi.update(s_ep, a_ep, r_ep)

    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

                 
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()
