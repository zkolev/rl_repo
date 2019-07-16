# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:53:37 2019

@author: zkolev
"""

import numpy as np
import matplotlib.pyplot as plt 


class Grid():
    def __init__(self, width , height, start):
        self.widht = width
        self.height = height
        self.i = start[0]
        self.j = start[1]
        
    def set(self, rewards, actions):
        ## Rewards are dict of the state (i, j): r
        ## Actions should be a dict of : (i, j)
        
        self.rewards = rewards
        self.actions = actions
        
    
    def set_state(self, s):
        self.i= s[0]
        self.j = s[1]
        
    def current_state(self):
        return(self.i, self.j)
        
    def is_terminal(self, s):
        return s not in self.actions
    
    def move(self, action):
        if action in self.actions[(self.i, self.j)]:
            
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'L':
                self.j -= 1
            elif action == 'R':
                self.j += 1
        return self.rewards.get((self.i, self.j),0)
    
    def undo_move(self, action):

        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'L':
            self.j += 1
        elif action == 'R':
            self.j -= 1
        
        assert(self.current_state() in  self.all_states())
    
    def game_over(self):
        return (self.i, self.j ) not in self.actions
         
    