# Hyperparameter-Optimization-for-Deep-Q-Networks
Final Project for COMS 6998 Deep Learning Systems Performance at Columbia University <br>

Collaborator: **In Wai Cheong** (https://www.github.com/InwaiCheong) <br>

References: https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch (heavily modified the DQN files from this repo)


## Projection Description
To measure the sensitivity of Deep Q-Networks on different tasks subject to learning rate, batch size, optimizer, target Q network update step size, discount factor, and other hyperparameters to identify the relationship between hyperparameters and efficient convergence to the optimal policy across different state/action regimes. <br>

## Methods Implemented
Random Search <br>
Successive Halving <br>
Bayesian Optimization

## Implementation Details
View report for in-depth details about our implementation.

## File Descriptions
The notebooks can be downloaded and ran as is. There are two notebooks each for Successive Halving and Random Search. One is the implementation and the other is visualization of the agent. Bayesianopt.ipynb includes the bayesian optimization implementation.

## Key Takeaways:
1. Even simple games are enormously sensitive to hyperparameter tuning.
2. Sample complexity of Deep RL is very high, and the reward signal is very sparse, implying that parametric methods that rely on information regarding obtained rewards (e.g. Bayesian Optimization) do not work very well. 
3. Given the scarcity of the reward signal in our examples, pseudo-evolutionary methods (such as successive halving approaches) actually work best. 
