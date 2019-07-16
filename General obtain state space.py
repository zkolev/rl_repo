# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:21:47 2019

@author: zkolev
"""


import numpy as np
from itertools import product


### Obtain all combinations of dice rolls for 5 dices

## Create placeholder for each combination of rolls
dices_arr = []

## Loop over all combinations
## and record them as list of np arrays 
for d1 in range(6):
    for d2 in range(6):
        for d3 in range(6):
            for d4 in range(6):
                for d5 in range(6):
                    
                    dice_roll_matrix = np.zeros((5,6))
                    itr_lst = [i for i in enumerate([d1,d2,d3,d4,d5])]
                    
                    for j in itr_lst:
                        dice_roll_matrix[j] = 1
                    
                    dices_arr .append(dice_roll_matrix)
                    
# Convert the list of 2d arrays to 3d array 
dices_arr  = np.array(dices_arr )
## Aggregate the faces of the dice 
## This way we get N rolls for each face
dices_arr = dices_arr.sum(axis = 1)
## Dedup the array 
dices_arr = np.unique(dices_arr, axis = 0)

#### Append the dice positions: 
#### Create all combinations between r= remaining rolls, dice combos and board state (checked out positions)
NumPositions = 2 
ingame_satates = [np.hstack((r, dices_arr[j,:], np.array(i))) 
        for r in range(3) 
            for j in range(dices_arr.shape[0]) 
                for i in product((0,1), repeat = NumPositions) if i != tuple([1]*NumPositions)]


### There are some special positions that must be considered 
### The common property of those states is that the dice roll is 0
### The starting state is all positions unchecked
### The terminal state is the oposite remaining rolls =3 , dice_roll = 5x0, positions = 1

special_states = [np.hstack((3, np.zeros((6)), np.array(i))) for i in product((0,1), repeat = NumPositions) ]

### 

all_states = np.vstack((ingame_satates,special_states))

all_states.shape