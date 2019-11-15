import numpy as np
from itertools import product
import random
from Environments.General.GeneralMandatory import GeneralGame
import pickle

##
## The action set depends on the state of the environment
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

## Define playgame function : 
## Play untill game over
    

def play_game_sarsa(env, P, Q, U, t = 1, eps = 0.3, ALPHA = 1, GAMMA = 1):

    ## P is the policy that our agent follows
    ## The policy may not be initialized however there should be a policy object 
    ## Q is the evaluation of the action value functio
    ## U is counter 
    
    ## NB! In order to work properly the Q and P must be initialized with the starting state !
    
    total_reward = 0
    
    ## During the initial state there is only one eligible action
    ## that the agent can take. 
    a = env.sample_random_action() 
    
    while not env.game_over:
        
        ## The policy is {state:action}
        ## Get the state of the game 
        
        s = env.game_state

        ## Perform the action and obtain the new state and the reward
        ## Also reiterate 
        
#        max_change = 0
#        old_qsa = Q[s][a]        

        s2, _ , r = env.perform_action(a)
        total_reward += r

        #################################################
        ### REITERATE THE LOGIC FOR STATE AND ACTIONS ###
        #################################################
        
        if env.game_over:
            
            ## Terminal State         
            Q[s2] = {}
            a2 = 'TerminalState'
            Q[s2][a2] = 0
            
        ## If we end in state that has been initiated
        ## Then sample random action and initiate Q 
        
        elif s2 in Q:
            
            a2 = max_dict(Q[s2])[0]
            a2 = env.eps_greedy(a2, eps = eps/t)
            
            if not a2 in Q[s2]:
                Q[s2][a2] = 0
                U[s2][a2] = 1
            
        ## If we end up in unseen state
        ## initiate the dict 
        else:
            
            ## Init the 
            Q[s2] = {}
            U[s2] = {}

            ## If there is no action in the action dictionary...
            ## ...perform random action based on the available action space: 
            ## Init the Q , U and Policy
            
            a2 = env.sample_random_action()
            P[s2] = a2
            Q[s2][a2] = 0
            U[s2][a2] = 1
            
            
        ############
        ## UPDATE ##
        ############
        
        alpha = ALPHA / U[s][a]
        U[s][a] += 0.00005
        
        try:
            Q[s][a] = Q[s][a] + alpha * (r + GAMMA*Q[s2][a2] - Q[s][a])
            a = a2
        except KeyError:
            print(s, s2, a, a2)
#        max_change = max(max_change, np.abs(old_qsa - Q[s][a]))
        
        ## Update Policy
        
        for s_new in Q.keys():
            new_a, max_val = max_dict(Q[s_new])
            P[s_new] = new_a
   
    return(P, Q, U, total_reward)

## Define simple play game according to a policy:
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
                        
            ## If the state s is not explored then take random action 
            if not s in P.keys():
                a = env.sample_random_action()
                
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

MyEnv = GeneralGame()
start_state = MyEnv.game_state
start_state_a = MyEnv.sample_random_action()
MyPolicy = {start_state:start_state_a}

## Init Q
## Q is the table where the action-value pairs will be evaluated
Q = {start_state:{start_state_a:0}}

## This is a counter that checks how many times 
## a state has been visited
U = {start_state:{start_state_a:1}}



## Start Play Loop 
play_iterations =  int(2e5)
save_iter = int(5e2)
saver_rounds = int(5e2)
save_loc = 'E:\\RL\\Results'
evaluated_rewards = []
unseen_states = []
training_rewards = []


## TRAINING LOOP ## 

for i in range(play_iterations):
    
    ## print progress: 
    if i%1000 == 0:
        print('Finished', i, 'iterations')
	
	## Play the game and update the Action-value table and the policy: 
    MyPolicy, Q, U, R = play_game_sarsa(GeneralGame(), MyPolicy, Q = Q, U = U, eps = 0.7)
	
	## Append the reward from the played episode: 
    training_rewards.append((i, R))

    ## For each save_iters play the game for saver_rounds times 
    ## If the agent is learning we expect the average score to increase
    ## This can further be developed by reevaluating the value function 
    
    if i%save_iter == 0:
        running_r = []
        running_unevals = []

        for r in range(saver_rounds):
            
            rr, rs = play_general(env = GeneralGame(), P = MyPolicy)
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