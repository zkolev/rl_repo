# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:13:02 2019

@author: zkolev
"""

import numpy as np
import random
from itertools import product 

## Define the state space: 

## After each roll we can either throw or not throw the dice
## For five dices there are 5**2 = 32 total actions that can be performed 
## The terminal state for round is the subscribe part where 
## The player must choose which position to subscribe 

## Define function that will yueld the available action space as 
## list of tuples.

def get_action_set(position_names):
    all_actions_set = []
    for d1 in range(6):
        for d2 in range(6):
            for d3 in range(6):
                for d4 in range(6):
                    for d5 in range(6):
                        for d6 in range(6):
                            a = (d1,d2,d3,d4,d5,d6)
                            if sum(a) <5:
                                all_actions_set.append(('Roll', a))
            
    for i in position_names:
        all_actions_set.append(('Checkout', i))
        
    return(all_actions_set)
                        

## General game class:
## Start general in 


class GeneralGame():
    def __init__(self,improvement_reward = 0.5, invalid_action_penalty = -100):
        
        self.improvement_reward = improvement_reward ## the reward the agent gets if after roll imrpoves the  potential reward 
        self.invalid_action_penalty = invalid_action_penalty 
        
        ## Game Controls:
        self.game_over = False
        
        ## Score and reward 
        ## The reward is the points introduced to the learning algorithm
        ## The score is the actual results that environment achieves 
        
        self.final_score = np.int32(0)
        self.last_score = np.int32(0)
        self.max_potential_score = 0
        
        self.dices = [Dice() for i in range(5)] ## Init Dices: 
        self.dices_agg = np.zeros(6,dtype = int)
        
        self.positions = [MandatoryPosition(1, 'One', True)
    						   ,MandatoryPosition(2, 'Two', True)
                         ,MandatoryPosition(3, 'Three', True)
                         ,MandatoryPosition(4, 'Four', True)
                         ,MandatoryPosition(5, 'Five', True)
                         ,MandatoryPosition(6, 'Six', True)
                               ]
        
        self.position_names = [p.get_position_name() for p in self.positions]
        
        ## Game controls 
        self.remaining_rolls = 3
        self.round = 0
        self.game_state = tuple(np.hstack((self.remaining_rolls, self.dices_agg, [int(position.get_checkout_state()[1]) for position in self.positions])))
        
        ## All actions:
        self.all_actions = get_action_set(position_names=self.position_names)


    ## This function prints the face of all dices
    ## It is not used in the actual game 
    ## it is for debugging only 
    def print_dice(self):
        all_dice_values =  [i.get_dice_value() for i in self.dices]        
        print('----------')
        print(*all_dice_values , sep =  '|' )
        print('----------')        
    
    
    ## Reset the state of the game 
    def reset(self):
        
        self.game_over = False
		
        # The final score updates only after checkout 
        # It represents the score from the scoreboard 
        self.final_score = np.int32(0) 
        
        ## The last score is the running reward of the episode
        self.last_score = np.int32(0)
        self.max_potential_score = 0
        
        # Reset dices
        for i in self.dices:
            i.reset()
            
        self.dices_agg = np.zeros(6,dtype = int)
        
        ## Reset the initial state of the positions
        for pos in self.positions:
            pos.reset_position()
        
        
        ## Game controls 
        self.remaining_rolls = 3
        self.round = 0
        self.game_state = tuple(np.hstack((self.remaining_rolls, self.dices_agg, [int(position.get_checkout_state()[1]) for position in self.positions])))
        

    
    def update_game_state(self):
        
        ## Returns tuple with the state 
        
        self.game_state = tuple(np.hstack((self.remaining_rolls, self.dices_agg, [int(position.get_checkout_state()[1]) for position in self.positions])))
        
    
    def print_game_state(self):
        
        ## This separates the different elements of the game state with '|' 
        ## the first section is the remaining rolls 
        ## the second section is the dice roll 
        ## the third section is the checked positions: 
        
        game_state_print = list(self.game_state)
        
        for i in [1,8]:
            game_state_print.insert(i, '|')
        
        print(game_state_print)

    
    def round_roll(self, dice_ix = []):
        
        assert(self.remaining_rolls > 0)
        
        # Roll the dice based on the faces type you want to roll
        # For example we want to roll 1 3s and 1 6s faces 

        if self.remaining_rolls == 3 or all([dix == 0 for dix in dice_ix]):
            dices_for_modification = range(5)
            
        else:
            ## Sanity check if there is more roll actions available 
            
            ## Create placehoder with indeces of the dices we will modify 
            dices_for_modification = []
            dices = [i.get_dice_value() for i in self.dices]
			
			
            ## For each position of dice_ix
            for k,i in enumerate(dice_ix):
 
            ## Check if it should be rolled
            ## IF no (0 value) then go to the next position
                if i == 0:
                    continue
            
            ## For each position != 0 
            ## Find the position of the dice with the face we need to roll
        
                val_pos = [j for j, v in enumerate(dices) if v == (k+1)]
                
        
                ### For all 
                while i > 0:
                    dices_for_modification.append(val_pos.pop(0))
                    i -= 1
            
        ### Roll The dice:
        for i in dices_for_modification:
            self.dices[i].roll()
        
        
        ## Get the dice roll 
        dice_roll =  [d.get_dice_value() for d in self.dices]
        
        ## Modify aggregated dices:
        dices_agg_tmp = np.zeros(6, dtype = int)
        
        ## The dice roll is aggregated per position onf the list 
        ## The first element represents face value i = 1 (indexed as 0 or [i-1])
        for i in dice_roll:
            dices_agg_tmp[i-1] += 1
            
        self.dices_agg = dices_agg_tmp

         ## Calculate the temporary score for each position
        for p in self.positions:
            p.calculate_score(dice_roll = dice_roll)
        
        ## This check if the roll action has improoved the 
        ## score on any of the positions and if so then 
        ## add reward of 1 
        
        ## if this is the first roll of the round 
        ## init the max_potential_score  after the roll 
        if not self.game_over:
            if  self.remaining_rolls == 3:
                self.max_potential_score = max([i.temporary_score for i in self.positions if i.unchecked])
                self.last_score = 0
            else:
                potential_after_roll = max([i.temporary_score for i in self.positions if i.unchecked])
                
                ## Compare if the roll has improoved the score
                ## If so ... assign reward: 
                ## If after the roll the potential is worse then 
                ## assign penalty 
                
                if self.max_potential_score < potential_after_roll:
                    self.last_score = self.improvement_reward 
                
                elif self.max_potential_score > potential_after_roll:
                    self.last_score = self.improvement_reward 
                
                else:
                    self.last_score = 0
                    
                #Update the potential
                self.max_potential_score = potential_after_roll 
        
        ## Decrease the remaining rolls: 
        self.remaining_rolls -= 1
        self.update_game_state()

   
    ## Checkout position is the action that yelds actual reward 
    def checkout_position(self, position_name):
                
        ## Get the name of all positions and the checkout space:
        position_checkout = [p.get_checkout_state() for p in self.positions]
        
        ## Iterate trhough all positions, obtained from previous loop
        ## extract the positions that have not been checked yet
        
        p_name = []
        for p_checkout in position_checkout:
            if p_checkout[1]:
                p_name.append(p_checkout[0])
        
        ## Assert that the position for checkout is not filled
        ## sanity check
        assert(position_name in p_name)
        
        ## Find the location of the position that we want to chekout in the scoreboard 
        position_name_ix  = [i[0] for i in position_checkout].index(position_name)
        
        ## Checkout Position 
        self.positions[position_name_ix].checkout()
        
        ## Update final score after checkout
        self.final_score += self.positions[position_name_ix].get_final_score()
        
        ## Update the last reward: 
        self.last_score = self.positions[position_name_ix].get_final_score()
        
        ## Reset the dice counter
        self.remaining_rolls = 3
        
        ## Reset the dice
        for dice in range(5):
            self.dices[dice].reset()
            
        ## Reset the aggregated dice board
        self.dices_agg = np.zeros(6, dtype = int)
        
        ## Check if all positions are filled
        ## IF so - update game over: 
        if all([not(p.unchecked) for p in self.positions]):
            self.game_over = True
        
        self.update_game_state()



    ## Print the score board:
    def get_scoreboard_state(self):
        scoreboard_state = [i.get_checkout_state() for i in self.positions]
        return(scoreboard_state)
    
    
    ## This function lists all posible states
    ## according to the configuration of the game 
    ## the dice rolls combinations are constant
    ## however the positions migth varry depending 
    ## on the complexity of the game
    ## this functionallity will be used in order to 
    ## train approximation methods 
    
    def get_state_space(self):
    
    # the state is an 1d array of shape 1 x 6 x 2 ^ <# positions >
    
        ## Create placeholder for each combination of rolls
        dices_arr = []
        num_position = len(self.positions)
        
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

        ingame_satates = [np.hstack((r, dices_arr[j,:], np.array(i))) 
                for r in range(3) 
                    for j in range(dices_arr.shape[0]) 
                        for i in product((0,1), repeat = num_position) if i != tuple([0]*num_position)]
        
        
        ### There are some special positions that must be considered 
        ### The common property of those states is that the dice roll is 0
        ### The starting state is all positions unchecked
        ### The terminal state is the oposite remaining rolls =3 , dice_roll = 5x0, positions = 1
        
        special_states = [np.hstack((3, np.zeros((6)), np.array(i))) for i in product((0,1), repeat = num_position) ]
        
        ### 
        
        state_space = np.vstack((ingame_satates,special_states))
        return(state_space)


    ## The action set depends on the state of the environment
    ## The following function lists all possible actions that 
    ## can be executed
    
    def get_actions_set(self):
        
        if self.game_over:
            return([('TerminalState',(0,0,0,0,0,0))])
		
		
            ## Get the state of the environmen
        s = self.game_state
        
        ## If this is preroll statet (start of the game or right after checkout):
        if s[0] == 3:
            return([('Roll',(0,0,0,0,0,0))])
            
        ## Ctreate placeholder for the action set 
        rd_action_set_raw = []
        
        for d1 in range(self.dices_agg[0]+1):
            for d2 in range(self.dices_agg[1]+1):
                for d3 in range(self.dices_agg[2]+1):
                    for d4 in range(self.dices_agg[3]+1):
                        for d5 in range(self.dices_agg[4]+1):
                            for d6 in range(self.dices_agg[5]+1):
                                rd_action_set_raw.append((d1,d2,d3,d4,d5,d6))
        
        ## Create the position rolls as numpy array 
        ## Action 0,0,0,0,0,0 = roll all dices 
        
        rd_action_set= np.array(rd_action_set_raw, dtype = int)
#        print(rd_action_set[np.sum(rd_action_set, axis = 1) < 5,:])
        rd_action_set = rd_action_set[np.sum(rd_action_set, axis = 1) < 5,:]
        

        ## Get the score board state:
        ## It is needed to determine which positions can be checked 
        ## get_scoreboard_state returns list of tuples with two elements
        ## the first element is the name of the position
        ## the second element is the online state of the position
        ## True means the the position is online 
        
        sb = self.get_scoreboard_state()
        
        ## dice rolls = 3 means that the round has just started
        ## The only action is to roll all five dices 
        
        ## A is a placeholder for all actions 
        ## A is defined as tuple of action and another element 
        ## When the action is Roll the second element is a tuple 
        ## of position faces 
        ## When the action is Checkout then the second element is 
        ## the id of the position to be ckeded out 
        
        ## B is a mirro ph for the online state of all environments
        
        A = []
        B = []
        
        for sb_i in sb:
            if sb_i[1]: # if the position is online 
                A.append(('Checkout',(sb_i[0]))) ## append the position as potential action 
                B.append(sb_i[0]) 
                
        ## Add dice roll if remaining rolls > 0
        if s[0] > 0:
            rws, _= rd_action_set.shape
            for rs in range(rws):
                A.append(('Roll', tuple(rd_action_set[rs,:].flatten())))
        
        ## If All positions are checked out then return emtpy list 
        ## This means that the game is in terminal state
        
        if all([not i for i in B]):
            A = []
        
        return(A)
        
#    def get_action_space(self):
        
        
    
    def perform_action(self, a):
        
        ### Verify that the action is in the possible actions space
        ### If not assign large negative reward: 
        
        A = self.get_actions_set()
        
        if any([j == a for j in A]):
    
            ### Apply action to the environment
            ### a should be dictionary with 1 element 
            ### The key of the dictionary should be the parameter 
            ### The value should be the type of the action
            ### There are 2 main types of actions - Checkout and Roll   
            
            Action_Key = a[1] # Roll / Checkout
            Action_Value = a[0] # Depends on the Action_Key 
            
            if Action_Value == 'Roll':
                self.round_roll(Action_Key)
            else:
                self.checkout_position(Action_Key)
        
        else:
            self.last_score = self.invalid_action_penalty
        
        # The reutrned game state is the state AFTER performing the action 
        return((self.game_state ,a,self.last_score))


    ## Define function with random action 
    ## That will implement eps greedy strategy
    
    def eps_greedy(self, a, eps = 0.1):
        ## Eps Soft implementation
        A = self.get_actions_set()
        p = np.random.random()
        if p < (1 - eps):
            return(a)
        else:
            return random.sample(A, k = 1)[0]
        
        
    ## Sample random action per state
    ## This is useful for benchmarking 
    def sample_random_action(self, only_eligible=True):
        ## The game can sample actions from the full action space 
        ## of only from the action space that leads to state change 
        ## this is controlled by the additional parameter of the method 
        
        if only_eligible:
            A = self.get_actions_set()
            return (random.sample(A, k = 1)[0])
        else:
            return (random.sample(self.all_actions, k = 1)[0])
        
        
    
### 
# The general game is all about checking out the different positions 
# Optimizing the score for each position is the primary objective of the game 
# Here each position is a placeholder for a combination of dice faces that have to be checked
# The number and type of positions comtrolls the complexity of the game 
# The original game contains 6 mandatory +  7 optional positions 
        
class Position():
    
    def __init__(self,position_name, is_mandatory):
        
        self.position_name = position_name
        self.unchecked = True
        self.final_score = np.int32(0)
        self.temporary_score = None
        self.is_mandatory= is_mandatory
    

    ## Checkout the position
    ## Fill the score in the score PH - final ph
    def checkout(self):
        if self.unchecked: 
            self.final_score = self.temporary_score
            self.unchecked=False
    
    def get_final_score(self):
        return(self.final_score)
    
    ## Reset the temporary score score calculation
    def reset(self):
        self.temporary_score = None
    
    def reset_position(self):
        
        self.unchecked = True
        self.final_score = np.int32(0)
        self.temporary_score = None
        
    def get_position_name(self):
        return(self.position_name)
    
    ## 
    def get_temp_score(self):
        if self.unchecked:
            return((self.position_name, self.temporary_score))
    
    ## Return Tuple (position name, checkout state)
    def get_checkout_state(self):
            return((self.position_name, self.unchecked))

### These will accomodate our mandatory positions 

class MandatoryPosition(Position):
    
    def __init__(self, n,position_name,is_mandatory, verbose = False):
        super(MandatoryPosition,self).__init__(position_name, is_mandatory)
        
        self.n = n
        self.verbose = verbose
        self.min_score = -2 * n
        self.max_score =  2 * n
        
    def calculate_score(self, dice_roll):
        
        ## If the2 position has been checked do not calculate score:
        if self.unchecked:
        
            ## Check how many ocurences of the position number occurs in the dice
        
            NumOc = sum([j in [self.n] for j in dice_roll])
            NumOc = max([1, NumOc])
            Score = (NumOc-3) * self.n
        
#            ## If reach General gain 50 points extra
#            if self.n == 5 and NumOc == 5:
#                Score += 50
        
            self.temporary_score = Score
        
## There are additional Position extensions that will be defined later 
        
 

### Create the dice class 
class Dice():
    def __init__(self):
        self.value = 0
    
    def roll(self):
        self.value = np.random.randint(low = 1, high = 7)
    
    def reset(self):
        self.value = 0
    
    def get_dice_value(self):
        return(self.value)
    