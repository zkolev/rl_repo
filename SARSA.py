import numpy as np
from itertools import product
import random
from Environments.GeneralGame.GeneralMini import GeneralGame
import pickle

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
    
    total_reward = 0
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
        
        if s in Q:
            
            a = max_dict(Q[s])[0]
            a = eps_greedy(a, A, eps = eps/t)
            
            if not a in Q[s]:
                Q[s][a] = 0
                U[s][a] = 1
            
        else:
            
            ## Init the 
            
            Q[s] = {}
            U[s] = {}

            ## If there is no action in the action dictionary...
            ## ...perform random action based on the available action space: 
            ## Init the Q , U and Policy
            
            a = random.sample(A, k=1)[0]
            P[s] = a
            Q[s][a] = 0
            U[s][a] = 1

        ## Perform the action and obtain the new state and the reward
        ## Also reiterate 
        
        max_change = 0
        old_qsa = Q[s][a]        

        s2, _ , r = perform_action(env, a)
        total_reward += r

        #################################################
        ### REITERATE THE LOGIC FOR STATE AND ACTIONS ###
        #################################################
        
        
        A2 = get_actions_set(env)
        
        if len(A2) == 0:
            ## Terminal State
            
            Q[s2] = {}
            a2 = 'TerminalState'
            Q[s2][a2] = 0
            

        elif s2 in Q:
            
            a2 = max_dict(Q[s2])[0]
            a2 = eps_greedy(a2, A2, eps = eps/t)
            
            if not a2 in Q[s2]:
                Q[s2][a2] = 0
                U[s2][a2] = 1
            
        else:
            
            ## Init the 
            Q[s2] = {}
            U[s2] = {}

            ## If there is no action in the action dictionary...
            ## ...perform random action based on the available action space: 
            ## Init the Q , U and Policy
            
            a2 = random.sample(A2, k=1)[0]
            P[s2] = a
            Q[s2][a2] = 0
            U[s2][a2] = 1
            
            
        ############
        ## UPDATE ##
        ############
        
        alpha = ALPHA / U[s][a]
        U[s][a] += 0.005
        
        try:
            Q[s][a] = Q[s][a] + alpha * (r + GAMMA*Q[s2][a2] - Q[s][a])
        except KeyError:
            print(s, s2, a, a2)
        max_change = max(max_change, np.abs(old_qsa - Q[s][a]))
        
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
update_count_sa[start_state][('Roll',(0, 0,0,0,0,0))]  = 1

## Start Play Loop 
play_iterations =  int(1e4)
save_iter = int(1e3)
saver_rounds = int(1e3)
save_loc = 'C:\\006 Learning\\RL\\SARSA RESULTS'
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
    MyPolicy, Q, update_count_sa, R = play_game_sarsa(GameInstance, MyPolicy, Q = Q, U = update_count_sa, eps = 0.7)
	
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
bm_iter = int(1e4)
save_loc = 'C:\\006 Learning\\RL\\SARSA RESULTS'

## Play the game:

for i in range(bm_iter):
    
    ## Start New Game Instance: 
    GameInstance = GeneralGame()
	
	## Play the game and update the Action-value table and the policy: 
    _, _, _, R = play_game_sarsa(GameInstance,MyPolicy_random, Q = Q_random, U = update_count_sa_random, eps = 1)
	
	## Append the reward from the played episode: 
    benchmark_rewards.append((i, R))


# Serialize bm results:
bm_out =  open(save_loc+ '\\baseline_reward.pickle', 'wb')
pickle.dump(benchmark_rewards , bm_out)
bm_out.close()