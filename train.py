#!/usr/bin/env python
# coding: utf-8

from collections import deque, namedtuple

from ddpg import DDPGAgent as Agent

from unityagents import UnityEnvironment
import numpy as np
import torch

def train(
        env,
        agent,
        n_episodes=1,
        max_time_steps=1000,
        target_score=0.5):
    """Training loop for a given agent in a given environment.

    Args:
        agent: instance of class Agent
        env: Unity environment
        n_episodes (int): maximum number of training episodes
        max_time_steps (int): maximum number of timesteps per episode
        target_score (float): target score at which to end training

    Side effects:
        Alters the state of `agent` and `env`.

    Returns:
        list: sum of all rewards per episode
    """
    scores = []
    scores_window = deque(maxlen=100)

    brain_name = env.brain_names[0]

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        score = 0

        agent.reset()

        for timestep in range(max_time_steps):

            # choose and execute actions
            actions = agent.act(states, noise=True)
            actions = np.clip(actions, -1, 1)

            env_info = env.step(actions)[brain_name]

            # observe state and reward
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # save action, obervation and reward for learning
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states

            score += max(rewards)

            agent.learn(timestep)

            if np.any(dones):
                break

        scores.append(score)
        scores_window.append(score)

        best_score = max(scores)
        window_mean = np.mean(scores_window)

        memory_size = len(agent.memory)
        num_learn = agent.learn_counter

        print(
            f"\rEpisode {i_episode}\tScore: {score:.2f}\tBest: {best_score:.2f}"
            f"\tMean: {window_mean:.2f}"
            f"\tTimesteps: {timestep}\tMem: {memory_size}\tLearn: {num_learn}")

        if window_mean >= target_score:
            print(f"\nTarget score reached in {i_episode-100:d} episodes!")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return scores

def main():

    env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64", no_graphics=True)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = Agent(state_size, action_size)

    scores = train(env, agent, n_episodes=1000)
    env.close()


if __name__ == "__main__":
    main()
