# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:38:18 2019

@author: zkolev
"""

import numpy as np
from itertools import product 
import random
from GeneralMini import GeneralGame

import pickle

## The action set depends on the state of the environment
## The following function lists all possible actions that 
## can be executed

def get_actions_set(env):
    
        ## Get the state of the environmnt
    s = env.get_game_state()
    
    ## Ctreate placeholder for the action set 
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
    
    A = []
    B = []
    
    if s[0] == 3:
        A.append(('Roll',(0,0,0,0,0,0)))
    else:
        for sb_i in sb:
            if sb_i[1]:
                A.append(('Checkout',(sb_i[0])))
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
        
## Apply action to the environment
### a should be dictionary with 1 element 
### The key of the dictionary should be the parameter 
### The value should be the type of the action
### There are 2 main types of actions - Checkout and Roll    

def perform_action(env, a):
    Action_Key = a[1]
    Action_Value = a[0]
    s = env.get_game_state()
    if Action_Value == 'Roll':
        env.round_roll(Action_Key)
        
        ## When you roll the dice you do not get scores
        r = 0
#        s = env.get_game_state()
    else:
        env.checkout_position(Action_Key)
        r = env.last_score
#        s = env.get_game_state()
    return((s,a,r))


## Define function with random action 
## That will implement eps greedy strategy
    

def eps_greedy(a, A, eps = 0.1):
    ## Eps Soft implementation
    
    p = np.random.random()
    if p < (1 - eps):
        return(a)
    else:
        return random.sample(A, k = 1)[0]

#########

def max_dict(d):
    ## Init phs:
    max_key = None
    max_val = float('-inf')
    for k,v in d.items():
        if v > max_val:
            max_key = k
            max_val = v
    
    return((max_key, max_val))

## Define function for the control problem
## Play untill game over
    
def play_game_mc(env, P, Q, U, t = 1, eps = 0.3, GAMMA = 1):

    ## P is the policy that our agent follows
    ## The policy may not be initialized however there should be a policy object 
    ## Q is the evaluation of the action value functio
    ## U is counter 
    
    total_reward = 0
    ## Create ph for the results of the episode
    ## E_Res will be a list of tuples 
    E_Res = []
    
    while not env.game_over:
        
        ## The policy is {state:action}
        ## Get the state of the game 
        s = env.get_game_state()
        
        ## If no policy is defined for that state then
        ## take random action from all possible states
        
        ## Get all possible game states
        A = get_actions_set(env)


        ## Check if there is an action according to the given policy
        ## If there is an action then perform the action 
        ## NB!:  Here the epsilon greedy policy should be implemented
        
        if s in P:
            
            a = P[s]
            a = eps_greedy(a, A, eps = eps/t)
    
            ## need to ini Q[s][a] for observations that are performed
            ## for first time
            if not a in Q[s]:
                Q[s][a] = 0
                U[s][a] = 0
            
        else:
            
            ## Init the placeholders
            Q[s] = {}
            U[s] = {}

            ## If there is no action in the action dictionary...
            ## ...perform random action based on the available action space: 
            ## Init the Q , U and Policy
            
            a = random.sample(A, k=1)[0]
            P[s] = a
            Q[s][a] = 0
            U[s][a] = 0

        ## Perform the action and obtain the new state and the reward
        ## Also reiterate     
        E_Res.append(perform_action(env, a))
            
    ############
    ## UPDATE ##
    ############
    
    running_reward = 0 # 

    for s_g, a_g, r in reversed(E_Res):
        
        running_reward = r + GAMMA * running_reward
        
        total_reward += r
        
        ## Initialize ph for unseen states 
        if s_g in Q:
            
            if not a_g in Q[s_g]:
                Q[s_g][a_g] = 0
                U[s_g][a_g] = 0
            
        else:
            
            ## Init the placeholders
            Q[s_g] = {}
            U[s_g] = {}

            ## Init ph for the action 
            Q[s_g][a_g] = 0
            U[s_g][a_g] = 0
        
        ## Update: 
        U[s_g][a_g] += 1
        Q[s_g][a_g] =  Q[s_g][a_g]*(1 - (1/U[s_g][a_g])) + running_reward * 1/U[s_g][a_g]   #calculate average 
    
    ## Update Policy
    
    for s_new in Q.keys():
        new_a, max_val = max_dict(Q[s_new])
        P[s_new] = new_a
   
    return(P, Q, U, total_reward)
    
    
    
    

## Define simple play game according to a given policy:
def play_general(env, P):
    
    ## This function is used to play the game acording to given policy
    ## If there is unexplored state then random action is taken 
        
        ## Keep track of the reward and the unseen actions
        unseen_states = 0
        total_reward = 0 
        
        while not env.game_over:
            
            ## The policy is {state:action}
            ## Get the state of the game 
            s = env.get_game_state()
            
            ## If no policy is defined for that state then
            ## take random action from all possible states
            
            ## Get all possible game states
            A = get_actions_set(env)
            
            ## If the state s is not explored then take random action 
            if not s in P.keys():
                a = random.sample(A, k=1)[0]
                
                ## Update the counter
                unseen_states +=1
            else:
                ## Play accordin to the ginve policy
                a = P[s]
                
            ## Do the action 
            s2, _ , r = perform_action(env, a)
            
            ## Update the total reward from the last actoin :
            total_reward +=  r
        
        ## When the game is over return the accumulated reward and the number of unseen actions 
        return total_reward , unseen_states


### INTERACTIVE PART ### 

## Initiate the objects of the play_game function
## Define initial values for the policy and the Value function: 

start_state = (3, '|', 0, 0, 0, 0, 0, 0,'|', True, True)
MyPolicy = {start_state:('Roll',(0,0,0,0,0,0))}

## Init Q
## Q is the table where the action-value pairs will be evaluated
Q  = {}
Q[start_state] = {}
Q[start_state][('Roll',(0, 0,0,0,0,0))] = 0

## This is a counter that checks how many times 
## a state has been visited
update_count_sa = {start_state:{}}
update_count_sa[start_state][('Roll',(0, 0,0,0,0,0))]  = 0

## Start Play Loop 
play_iterations =  int(1e5)
save_iter = int(5e3)
saver_rounds = int(1e3)
save_loc = 'C:\\006 Learning\\RL\\MC RESULTS'
evaluated_rewards = []
unseen_states = []
training_rewards = []


## TRAINING LOOP ## 

for i in range(play_iterations):
    ## Start New Game Instance: 
    GameInstance = GeneralGame()
    
    ## print progress: 
    if i%1000 == 0:
        print('Finished', i, 'iterations')
	
	## Play the game and update the Action-value table and the policy: 
    
    MyPolicy, Q, update_count_sa, R = play_game_mc(GameInstance, MyPolicy, Q = Q, U = update_count_sa, eps = 0.7)
	
	## Append the reward from the played episode: 
    training_rewards.append((i, R))

    ## For each save_iters play the game for saver_rounds times 
    ## If the agent is learning we expect the average score to increase
    ## This can further be developed by reevaluating the value function 
    
    if i%save_iter == 0:
        running_r = []
        running_unevals = []

        for r in range(saver_rounds):
            
            rr, rs = play_general(env = GeneralGame(),P = MyPolicy)
            running_r.append(rr)
            running_unevals.append(rs)
    
        evaluated_rewards.append(running_r)
        unseen_states.append(running_unevals)


### Serialize data: 

# Serialize training rewards
tr_rew_out =  open(save_loc+ '\\training_rewards.pickle', 'wb')
pickle.dump(training_rewards, tr_rew_out)
tr_rew_out.close()
# Serialize evaluation rewards: 
ev_rew_out =  open(save_loc+'\\evaluated_rewards.pickle', 'wb')
pickle.dump(evaluated_rewards, ev_rew_out)
ev_rew_out.close()

# Serialize the policy 
Policy =  open(save_loc+ '\\policy.pickle', 'wb')
pickle.dump(MyPolicy, Policy )
Policy .close()

# Serialize the action value: 
action_value = open(save_loc+ '\\action_value.pickle', 'wb')
pickle.dump(Q, action_value)
action_value.close()

##########################
## BASELINE FUNCTION :  ##
##########################

## This is used for evaluation of our learning process
## against the random actions : 

# Init 
start_state = (3, '|', 0, 0, 0, 0, 0, 0,'|', True, True)
MyPolicy_random = {start_state:('Roll',(0,0,0,0,0,0))}

## Init Q
## Q is the table where the action-value pairs will be evaluated
Q_random = {}
Q_random[start_state] = {}
Q_random[start_state][('Roll',(0, 0,0,0,0,0))] = 0

## This is a counter that checks how many times 
## a state has been visited
update_count_sa_random = {start_state:{}}
update_count_sa_random[start_state][('Roll',(0, 0,0,0,0,0))]  = 1



benchmark_rewards = []
bm_iter = int(1e3)
save_loc = 'C:\\006 Learning\\RL\\MC RESULTS'

## Play the game:

for i in range(bm_iter):
    
    ## Start New Game Instance: 
    GameInstance = GeneralGame()
	
	## Play the game and update the Action-value table and the policy: 
    _, _, _, R = play_game_mc(GameInstance,MyPolicy_random, Q = Q_random, U = update_count_sa_random, eps = 1)
	
	## Append the reward from the played episode: 
    benchmark_rewards.append((i, R))


# Serialize bm results:
bm_out =  open(save_loc+ '\\baseline_reward.pickle', 'wb')
pickle.dump(benchmark_rewards , bm_out)
bm_out.close()