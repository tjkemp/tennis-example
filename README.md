# Solving Unity Tennis Environment

[image1]: https://raw.githubusercontent.com/Unity-Technologies/ml-agents/0.4.0/docs/images/tennis.png "Trained Agent"


### Introduction

This project implements a deep reinforcement learning policy gradient algorithm Deep Deterministic Policy Gradient (DDPG) which can operate over continuous action spaces, and trains it to play tennis.

### The environment

The algorithm is trained against environment similar to [Tennis](https://github.com/Unity-Technologies/ml-agents/tree/0.4.0/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of both agent is to keep the ball in play.

### Solving the environment

The agent is modeled as a Deep Deterministic Policy Gradient (DDPG) agent (see the [paper](https://arxiv.org/pdf/1509.02971.pdf)).

State space is continuous and the size 2, action space is continuous space size per agent is 2. Each agent receives its own, local observation and same agent is simultaneously trained to play both agents through self-play. 

To environment is considered solved when the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

### The trained agent

The included agent finished training after 112 episodes with a best score of 5.20.

As the agent gets reward for keeping the ball in the air, it learns a cooperative, safe playing style. The trained agent seems not to use vertical movements at all and strongly prefers horizontal movement. I think it is because vertical movement brings more uncertainties to the trajectories and in a cooperative game that is not desirable.

An interesting example of a safe strategy the agent learned, is when both rackets wait the ball at the very ends of their respective sides. The left starts going fast toward the ball when it reaches a certain height on its side of the field. The right racket also waits for the ball at its side, but instead of moving towards the ball, it holds there to catch the ball and to stop its momentum completely. Then the agent carefully gives a small nudge to the ball to pass it back to the left side.

## Getting Started

1. Download and uninstall the environment.

```bash
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
unzip Tennis_Linux.zip
```

2. Install requirements

First, create and activate a python environment, and then install the requirements.

```bash
pip install -r requirements.txt
```

3. Test the set up by playing Tennis with an agent which does random moves:

```bash
python random_play.py
```

4. See a trained agent play the game:

```bash
python play.py
```

5. Train the agent

You can train the agent my simply running `train.py`.

```
$ python train.py

Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: , 
Episode 1       Score: 0.00     Best: 0.00      Mean: 0.00      Timesteps: 13   Mem: 28 Model updates: 0
...
Episode 50      Score: 0.00     Best: 0.20      Mean: 0.02      Timesteps: 13   Mem: 1888       Model updates: 689
Episode 60      Score: 0.10     Best: 0.30      Mean: 0.03      Timesteps: 33   Mem: 2436       Model updates: 963
Episode 70      Score: 0.00     Best: 0.40      Mean: 0.03      Timesteps: 13   Mem: 3048       Model updates: 1269
Episode 80      Score: 0.30     Best: 1.10      Mean: 0.08      Timesteps: 70   Mem: 4918       Model updates: 2204
Episode 90      Score: 0.80     Best: 1.20      Mean: 0.13      Timesteps: 167  Mem: 7150       Model updates: 3320
Episode 100     Score: 1.60     Best: 5.20      Mean: 0.31      Timesteps: 337  Mem: 14832      Model updates: 7161
Episode 101     Score: 3.10     Best: 5.20      Mean: 0.34      Timesteps: 605  Mem: 16044      Model updates: 7767
Episode 102     Score: 0.10     Best: 5.20      Mean: 0.34      Timesteps: 53   Mem: 16152      Model updates: 7821
Episode 103     Score: 4.50     Best: 5.20      Mean: 0.38      Timesteps: 893  Mem: 17940      Model updates: 8715
Episode 104     Score: 0.40     Best: 5.20      Mean: 0.39      Timesteps: 89   Mem: 18120      Model updates: 8805
Episode 105     Score: 0.60     Best: 5.20      Mean: 0.39      Timesteps: 130  Mem: 18382      Model updates: 8936
Episode 106     Score: 0.00     Best: 5.20      Mean: 0.39      Timesteps: 14   Mem: 18412      Model updates: 8951
Episode 107     Score: 1.70     Best: 5.20      Mean: 0.41      Timesteps: 337  Mem: 19088      Model updates: 9289
Episode 108     Score: 1.60     Best: 5.20      Mean: 0.42      Timesteps: 319  Mem: 19728      Model updates: 9609
Episode 109     Score: 3.80     Best: 5.20      Mean: 0.46      Timesteps: 740  Mem: 21210      Model updates: 10350
Episode 110     Score: 3.10     Best: 5.20      Mean: 0.49      Timesteps: 607  Mem: 22426      Model updates: 10958
Episode 111     Score: 0.00     Best: 5.20      Mean: 0.49      Timesteps: 13   Mem: 22454      Model updates: 10972
Episode 112     Score: 1.00     Best: 5.20      Mean: 0.50      Timesteps: 205  Mem: 22866      Model updates: 11178

Target score reached in 112 episodes!
```


## Licenses and acknowledgements

- The code is based on [Udacity Deep Reinforcement Learning nanodegree](https://github.com/udacity/deep-reinforcement-learning/) materials and is thus continued to be licensed under [MIT LICENSE](LICENSE).

## Author

- [tjkemp](https://github.com/tjkemp)
