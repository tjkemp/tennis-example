#!/usr/bin/env python
# coding: utf-8

from unityagents import UnityEnvironment
import numpy as np
import torch

from ddpg import DDPGAgent as Agent

def play(env, agent):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    for i in range(2):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        num_agents = len(env_info.agents)
        action_size = brain.vector_action_space_size
        scores = np.zeros(num_agents)

        while True:
            actions = agent.act(states, noise=False)
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

    # get action_size and state_size
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = Agent(state_size, action_size)
    agent.actor_local.load_state_dict(torch.load('files/checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('files/checkpoint_critic.pth'))

    play(env, agent)

    env.close()


if __name__ == "__main__":
    main()
