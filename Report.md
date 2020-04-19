# Report

## Learning algorithm

The implemented learning algorithm to solve the Reacher Unity environment is the Deep Deterministic Policy Gradient (DDPG) algorithm, first introduced by [research paper](https://arxiv.org/abs/1509.02971).

In the paper, they present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces.

The actor learns to approximate the best action for each state. The critic learns how to evaluate the action-value by using actions from the actor.

**Implementation details:**
- The actor model has two hidden layers, of size 128 and 64
- The critic model has three hidden layers, of size 64, 32 and 32
- Adam optimizer with learning rate 3e-3 is used for both models
- Batch size is 512
- Tau is 2e-1
- Replay buffer size is 1e5
- No weight decay

The learning rate is, and tau have larger values than in training Reacher environment in the previous project.

In contrast to the chosen parameters, the paper uses a larger layer sizes, and batch size of 64 for low dimensional problems.

## Plot of rewards

![Score plot](files/scores.png?raw=true "Plot of rewards")

The agent receives an average reward (over 100 episodes) of 0.5 after 112 episodes.

The environment can be considered solved.

## Ideas for future work

- I'd like to use the algorithm to solve other environments as well
- MADDPG approach would be a interesting thing to try out
- The paper suggests using Batch Normalization in order to normalize each dimension across the samples in minibatch and perform well across different tasks.
