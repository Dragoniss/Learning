# coding :utf-8

__author__ = 'zhenhang.sun@gmail.com'
__version__ = '1.0.0'

import gym
from  dog_sheep import *
from ddpg import Agent

if __name__ == '__main__':
    env = Dog_sheep()
    # env = gym.make('Pendulum-v0')
    env.reset()
    env.render()

    params = {
        'env': env,
        'gamma': 0.99,
        'actor_lr': 5e-5,
        'critic_lr': 5e-5,
        'tau': 0.02,
        'capacity': 10000,
        'batch_size': 32,
    }

    agent = Agent(**params)

    for episode in range(100):
        s0 = env.reset()
        episode_reward = 0
        # print(s0)
        for step in range(1500):
            env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            agent.put(s0, a0, r1, s1)
            episode_reward += r1
            if done:
                break
            s0 = s1
            # print("hahahaha")
            agent.learn()
            # print("qaq")
        print(episode, ': ', episode_reward)