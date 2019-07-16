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
    
def play_game_mc(env, P, Q, U, t = 1, eps = 0.3, GAMMA = 1, update_policy = True):

    ## P is the policy that our agent follows 
    ## The policy may not be initialized however there should be a policy object 
    ## Q is the evaluation of the action value functio
    ## U is counter that accounts for how many times state-action pair is visited
    
    
    ## Create ph for the results of the episode
    ## E_Res will be a list of tuples 
    episode_results = []
    
    while not env.game_over:
        
        ## The policy is {state:action}
        ## Get the state of the game 
        s = env.game_state
        
        ## If no policy is defined for that state then
        ## take random action from all possible states
        
        ## Get all possible game states
        A = env.get_actions_set()


        ## Check if there is an action according to the given policy
        ## If there is an action then perform the action 
        
        if s in P:
            
            a = P[s]
            a = eps_greedy(a, A, eps = eps/t)
            
        else:            
            a = random.sample(A, k=1)[0]
            
        ## Perform the action and obtain the new state and the reward
        ## Also reiterate    
        ## The perform_action method returns s-prime 
        ## we discard it and pack back in the tuple s 
        _, a, r = env.perform_action(a)
        episode_results.append((s,a,r))
           
    ############
    ## UPDATE ##
    ############
    
    ## Perform monte carlo evaluation 
    ## Aggregate the rewards backwards: 
    
    running_reward = 0

    for S, A, R in reversed(episode_results):
        
        ## Update the running reward
        running_reward = R + GAMMA * running_reward
        
        ## Initialize ph for unseen states and actions 
        ## In the action value dict: 
        if S in Q: #if the state is listed, but the action not 
            
            if not A in Q[S]:
                Q[S][A] = 0
                U[S][A] = 0
            
        else: #if the state (and from this the action as well) is not visited
            
            ## Init the placeholders for the state
            Q[S] = {}
            U[S] = {}

            ## Init ph for the action 
            Q[S][A] = 0
            U[S][A] = 0
        
        
        ## Update the value function for the Policy: 
        ## Use running mean 
        
        U[S][A] += 1
        Q[S][A] =  Q[S][A]*(1 - (1/U[S][A])) + running_reward * 1/U[S][A]   #calculate average 
    
    ## Update Policy
    
    if update_policy:
        
        for s_new in Q.keys():
            new_a, max_val = max_dict(Q[s_new])
            P[s_new] = new_a
   
    return(P, Q, U, env.final_score)
    


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
            s = env.game_state
            
            ## If no policy is defined for that state then
            ## take random action from all possible states
            
            ## Get all possible game states
            A = env.get_actions_set()
            ## If the state s is not explored then take random action 
            if not s in P.keys():
                a = random.sample(A, k=1)[0]
                
                ## Update the counter
                unseen_states +=1
            else:
                ## Play accordin to the ginve policy
                a = P[s]
                
            ## Do the action 
            s2, _ , r = env.perform_action(a)
            
            ## Update the total reward from the last actoin :
            total_reward +=  r
        
        ## When the game is over return the accumulated reward and the number of unseen actions 
        return env.final_score , unseen_states


### INTERACTIVE PART ### 

## Initiate the objects of the play_game function
## Define initial values for the policy and the Value function: 



## Init phs: 
P, Q, U = {}, {}, {}

## Start Play Loop 
play_iterations =  int(5e4)
save_iter = int(1e3)
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
    
    P, Q, U, R = play_game_mc(GameInstance, P = P, Q = Q, U = U, eps = 0.7)
	
	## Append the reward from the played episode: 
    training_rewards.append((i, R))

    ## For each save_iters play the game for saver_rounds times 
    ## If the agent is learning we expect the average score to increase
    ## This can further be developed by reevaluating the value function 
    
    if i%save_iter == 0:
        running_r = []
        running_unevals = []

        for r in range(saver_rounds):
            
            rr, rs = play_general(env = GeneralGame(),P = P)
            running_r.append(rr)
            running_unevals.append(rs)
    
        evaluated_rewards.append(running_r)
        unseen_states.append(running_unevals)


### Serialize data:         
with open(save_loc+ '\\training_rewards.pickle', 'wb') as tr_rew_out: 
    pickle.dump(training_rewards, tr_rew_out)

# Serialize training rewards
with open(save_loc+ '\\training_rewards.pickle', 'wb') as tr_rew_out:
    pickle.dump(training_rewards, tr_rew_out)

# Serialize evaluation rewards: 
with open(save_loc+'\\evaluated_rewards.pickle', 'wb') as ev_rew_out:
    pickle.dump(evaluated_rewards, ev_rew_out)

# Serialize the policy 
with open(save_loc+ '\\policy.pickle', 'wb') as Policy:
    pickle.dump(P, Policy )

# Serialize the action value: 
with open(save_loc+ '\\action_value.pickle', 'wb') as action_value:
    pickle.dump(Q, action_value)

##########################
## BASELINE FUNCTION :  ##
##########################

## This is used for evaluation of our learning process
## against the random actions : 



## Init phs: 
P_bm = {}


benchmark_rewards = []
bm_iter = int(1e3)
save_loc = 'C:\\006 Learning\\RL\\MC RESULTS'

## Play the game:

for i in range(bm_iter):
	
	## Play the game and update the Action-value table and the policy: 
    R, _ = play_general(GeneralGame() ,P = P_bm)
	
	## Append the reward from the played episode: 
    benchmark_rewards.append((i, R))


# Serialize bm results:
bm_out =  open(save_loc+ '\\baseline_reward.pickle', 'wb')
pickle.dump(benchmark_rewards , bm_out)
bm_out.close()