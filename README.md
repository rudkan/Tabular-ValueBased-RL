# Tabular Reinforcement Learning (Stochastic Windy Gridworld)

## Project Overview

This repository contains the implementation of various algorithms studied in tabular, value-based Reinforcement Learning (RL) for Assignment 1 of the Reinforcement Learning course. The focus of this assignment is on understanding the principles of **Dynamic Programming** (DP), **Q-learning**, **SARSA**, and **N-step Q-learning**, using a stochastic windy grid world environment.

## Structure of the Repository

The code provided in this repository covers different sections of the assignment as outlined below:

1. **Dynamic Programming (DP)**:
   - Implementation of Q-value iteration to compute the optimal policy in an environment where the model is known.
   
2. **Exploration (Q-learning)**:
   - Implementation of ϵ-greedy and Boltzmann (Softmax) exploration techniques for learning optimal policies when the model is not known.
   
3. **Back-up: On-policy vs Off-policy (SARSA)**:
   - Comparison between on-policy (SARSA) and off-policy (Q-learning) methods for learning.
   
4. **Back-up: Depth of Target (N-step Q-learning & Monte Carlo)**:
   - Comparison of different depths of back-up targets using 1-step, N-step, and Monte Carlo methods.

## Files in the Repository

- **`Agent.py`**: Contains the base agent class and exploration methods such as ϵ-greedy and softmax policies.
- **`DynamicProgramming.py`**: Implements Q-value iteration for dynamic programming.
- **`Q_learning.py`**: Implements the Q-learning algorithm.
- **`SARSA.py`**: Implements the SARSA algorithm.
- **`Nstep.py`**: Implements the N-step Q-learning algorithm.
- **`MonteCarlo.py`**: Implements the Monte Carlo method for reinforcement learning.
- **`Environment.py`**: Defines the stochastic windy grid world environment.
- **`Experiment.py`**: Contains code for running and comparing the experiments.
- **`Helper.py`**: Provides utility functions for plotting and smoothing results.

## Methodology

1. **Dynamic Programming**: 
   - The Q-value iteration algorithm is implemented to solve the gridworld environment by finding the optimal policy using Bellman's equation.

2. **Q-learning**:
   - We implement both ϵ-greedy and softmax exploration policies for Q-learning. Different values of exploration parameters ϵ and temperature τ were compared.

3. **SARSA**:
   - This section focuses on the comparison between on-policy (SARSA) and off-policy (Q-learning) algorithms. Both algorithms were tested with different learning rates to assess performance.

4. **N-step & Monte Carlo Methods**:
   - N-step Q-learning and Monte Carlo methods are implemented to evaluate the effect of different back-up depths in RL learning.

## Experiments

- The experiments were conducted on the **Stochastic Windy Gridworld** environment, where the agent moves through a 10x7 grid with a windy environment affecting its movements.
- Different settings were explored to observe the performance of each algorithm:
  - Dynamic programming converges faster and guarantees optimality.
  - Q-learning and SARSA balance exploration and exploitation with the trade-off between convergence speed and policy optimality.
  - N-step methods and Monte Carlo highlight the impact of depth on performance.

## Results & Interpretation

1. **Dynamic Programming** achieved faster convergence to the optimal policy due to complete knowledge of the environment.
   
2. **Q-learning vs SARSA**: Q-learning performed slightly better than SARSA in terms of final performance, but both algorithms showed similar trends.
   
3. **N-step vs Monte Carlo**: Shorter back-ups (1-step) performed better than longer back-ups (Monte Carlo), which suffered from higher variance.

## How to Run the Code

1. Install Python 3 and the required libraries:
   ```bash
   pip install numpy matplotlib scipy

2. Run the environment and observe the agent's actions:
   ```bash
    python Environment.py
   
3. To run Dynamic Programming, execute:
   ```bash
    python DynamicProgramming.py
   
4. To run Q-learning, SARSA, or other experiments:
    ```bash
    python Q_learning.py
    python SARSA.py
    python Nstep.py
    python MonteCarlo.py

5. The experiments are configured in Experiment.py and results will be plotted after execution.


## Conclusion
This project demonstrates the basic principles of tabular reinforcement learning methods. Through different algorithms and experiments, the trade-offs between exploration, policy optimality and computational efficiency are explored in a stochastic environment.
