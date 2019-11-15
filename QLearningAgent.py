# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:55:13 2019

@author: zhivk
"""

import collections
from Environments.General.GeneralMandatory import GeneralGame

## CONSTANTS 
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        self.env = GeneralGame()
        self.state = self.env.game_state
        self.Qtable = collections.defaultdict(float)
        
    def sample_env(self):
        a = self.env.sample_random_action()
        s = self.state
        next_s, _, r= self.env.perform_action(a)
        self.state = self.env.reset() if self.env.game_over else next_s
        return(s, a, r, next_s)
    
    def best_value_and_action(self, state):
        top_a, top_v = None, None
        for action in self.env.get_actions_set():
            action_value = self.Qtable.get((state, action))
            if action_value is None:
                action_value = -5
            if top_v is None or top_v < action_value:
                top_v = action_value
                top_a = action
        return(top_v, top_a)
        
    def value_update(self, s, a ,r , next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v
        old_val = self.Qtable[(s,a)]
        self.Qtable[(s,a)] = old_val * (1-ALPHA) +  new_val * (ALPHA)

    def play_episode(self):
        total_reward = 0
        self.env.reset()
        state = self.env.game_state
        while not(self.env.game_over):
            _, action = self.best_value_and_action(state)
            new_state, _, r = self.env.perform_action(action)
            total_reward += r
            state = new_state
        return(total_reward, self.env.final_score)

if __name__ == "__main__":
    myEnv = GeneralGame()
    agent = Agent()
#    writer 
    rew = []
    best_rew = []
    iter_no = 0
    best_reward = 0
    while True: 
        iter_no += 1
        print('Iteration', iter_no)
        agent.env.reset()
        s, a ,r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)
        reward = 0
        score = 0
        for _ in range(TEST_EPISODES):
            running_reward, running_score = agent.play_episode()
            reward += running_reward
            reward  /= TEST_EPISODES
            score  += running_score
            if reward > best_reward:
                best_reward = reward
        rew.append(score)
        best_rew.append(best_reward)
        if iter_no > int(1e3):
            break

#
#
agent.Qtable

#agent.value_update(s,a,r,next_s)
#agent.best_value_and_action(s)
#
#t = agent.Qtable.get((s,a))
