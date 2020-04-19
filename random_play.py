#!/usr/bin/env python
# coding: utf-8

from unityagents import UnityEnvironment
import numpy as np


def print_env_info(env):

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


def random_play(env):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    for i in range(5):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        num_agents = len(env_info.agents)
        action_size = brain.vector_action_space_size
        scores = np.zeros(num_agents)

        while True:
            actions = np.random.randn(num_agents, action_size)
            actions = np.ones((num_agents, action_size))
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards
            states = next_states

            if np.any(dones):
                break
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


def main():
    env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
    print_env_info(env)
    random_play(env)
    env.close()


if __name__ == "__main__":
    main()
