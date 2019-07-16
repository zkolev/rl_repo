import numpy as np
from itertools import product 
import random
from General import GeneralGame


#x = GeneralGame()

### Define the dice combinations outside the function
### In order  not to compute them every time

## Roll & Checkout actions: 

def get_actions_set(env):
    
    ## Get roll dice action set:
    s = env.get_game_state()
    
    
    rd_action_set_raw = []
    for d1 in range(env.dices_agg[0]+1):
        for d2 in range(env.dices_agg[1]+1):
            for d3 in range(env.dices_agg[2]+1):
                for d4 in range(env.dices_agg[3]+1):
                    for d5 in range(env.dices_agg[4]+1):
                        for d6 in range(env.dices_agg[5]+1):
                            rd_action_set_raw.append((d1,d2,d3,d4,d5,d6))
                        
    rd_action_set= np.array(rd_action_set_raw, dtype = int)[1:,]
    
#    state_value = tuple(s.keys())[0]
    ## Get the score board state:
    sb = env.get_scoreboard_state()
    
    ## If # dice rolls = 3 means that the round has just started
    ## The only action is to roll all five dices 
    
    A = {}
    if s[0] == 3:
        A = {(0,0,0,0,0,0):'Roll'}
    else:
        for sb_i in sb:
            if sb_i[1]:
                A.update({sb_i[0]:'Checkout'})
        
        ## Add dice roll if remaining rolls > 0
        if s[0] > 0:
            rws, _= rd_action_set.shape
            for rs in range(rws):
                A.update({tuple(rd_action_set[rs,:].flatten()):'Roll'})
    
    return(A)
        
### Here implement the initial policy as a function          



## Apply action to the environment
## A should be dictionary with 1 element 
### The key of the dictionary should be the parameter 
### The value should be the type of the action
### There are 2 main types of actions - checkout and diceRoll    
def perform_action(env, a):
    Action_Key = list(a.keys())[0]
    Action_Value = a.get(Action_Key)
    
    if Action_Value == 'Roll':
        env.round_roll(Action_Key)
        ## When you roll the dice you do no t get scores
        r = 0
        s = env.get_game_state()
    else:
        env.checkout_position(Action_Key)
        r = env.last_score
        s = env.get_game_state()
    return((s,a,r))





## Define playgame function : 
## Play untill game over
    
def play_game(env, pol = None):
    ## The policy is {state:action}
    ## Get the state of the game 
    s = env.get_game_state()
    s_and_r = [(s,{(0,0,0,0,0,0): 'Roll'}, 0)]    
    
    while not env.game_over:
        ## If no policy is defined for that state then
        ## take random action from all possible states
        if pol == None: 
           ## Get all possible game states
           A = get_actions_set(env)
           ## Select random action
           a = random.sample(A.keys(), k=1)[0]
           game_move= {a:A[a]}
        else:
            try:
                game_move = pol[s]
                              
            except KeyError:
                print('State', s, 'has not been explored')
                ## Perform random action: 
                A = get_actions_set(env)
                a = random.sample(A.keys(), k=1)[0]   
                game_move= {a:A[a]}
                
        tpl = perform_action(env, game_move)
        s = env.get_game_state()
        s_and_r.append(tpl)
   
    return(s_and_r)
    
x = GeneralGame()
play_game(x)

MyPolicy = {(3, '|', 0, 0, 0, 0, 0, 0,'|', True):{(0, 0,0,0,0,0): 'Roll'}}
V  = {(3, '|', 0, 0, 0, 0, 0, 0,'|', True):0}


### INIT HYPER PARAMETERS:


for i in range(int(1e5)):
    
    ## Init the game
    GenGame = GeneralGame()
    ## GameResults 
    s_a_r = play_game(GenGame, MyPolicy)
    
    for t in range(len(s_a_r)-1):
        s0,_,_ = s_a_r[t]
        s1,a,r = s_a_r[t+1]
        ###
        
        if not s1 in V:
            V[s1] = 0
        ## update policy: 
        if not s0 in MyPolicy:
            MyPolicy[s0] = a
        ## Calculate value function: 
        ## V[s]  = V[s0] + ALPHA*(r0 + Gamma*V[s1])
        
        V[s0] = V[s0] + (r + V[s1] - V[s0])


len(V)

MyPolicy[(2, '|', 3, 0, 1, 1, 0, 0, '|', True)]
V[(3, '|', 0, 0, 0, 0, 0, 0,'|', True)]