#!/usr/bin/env python
# coding: utf-8

from collections import deque, namedtuple

from unityagents import UnityEnvironment
import numpy as np
import torch

def train(
        env,
        agent,
        n_episodes=10,
        max_time_steps=300,
        target_score=30.):
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

        for t in range(max_time_steps):

            # choose and execute an action
            actions = agent.act(states)
            actions = np.clip(actions, -1, 1)

            env_info = env.step(actions)[brain_name]

            # observe state and reward
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            ended = env_info.local_done

            # save action, obervation and reward for learning
            agent.step(states, actions, rewards, next_states, ended)
            states = next_states

            score += np.mean(rewards)

            if np.any(ended):
                print("Episode ended before limit {max_time_steps}")
                break

        scores.append(score)
        scores_window.append(score)

        max_score = np.max(scores)

        print(f"\rEpisode {i_episode}\tAvg score: {score:.2f}\tMax score: {max_score:.2f}")

        if np.mean(scores_window) >= target_score:
            print(f"\nTarget score reached in {i_episode-100:d} episodes!")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return scores

def main():
    env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64", no_graphics=True)
    scores = train(env)
    env.close()


if __name__ == "__main__":
    main()
