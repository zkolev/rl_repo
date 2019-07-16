# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:45:12 2019

@author: zkolev
"""

from GeneralMini import GeneralGame
import tensorflow as tf
import numpy as np


gameinst = GeneralGame()
D = len(gameinst.game_state)
K = len(gameinst.all_actions)


## D: number of state features 
## K: the size of the action space:

class PolicyModel:
    def __init__(self, D, K ):  
        
        self.state_features = tf.placeholder(dtype = tf.float32, shape = (None, D), name = 'StateFeatures')
        self.actions = tf.placeholder(dtype = tf.int32, shape = (None,), name = 'Actions')
        self.advantages = tf.placeholder(dtype = tf.float32, shape = (None,1), name = 'Advantages' )
        
        ## SETUP THE GRAPH:
        self.l1 = tf.keras.layers.Dense(units = 40, activation = tf.nn.sigmoid).apply(self.state_features)
        self.l2 = tf.keras.layers.Dense(units = 20, activation = tf.nn.sigmoid).apply(self.l1)
        self.lfinal =  tf.keras.layers.Dense(units = K, activation=tf.nn.softmax).apply(self.l2) ## Our actual predict
        
        ## Cost function: 
        
        ## This is log pi(a|S, Theta) ~ log of probability of action given state
        selected_probs = tf.log(
                            tf.reduce_sum(
                                    self.lfinal * tf.one_hot(self.actions, K),
                                    reduction_indices = [1]
                                    )
                            )
        ## Cost function
        cost = -tf.reduce_sum(self.advantages * selected_probs)
        
        ## TrainOp
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
        
    
    def set_session(self, session):
        self.session = session     
        
        
    def partial_fit(self, X, actions, advantages):
        # Transform in 2d array 
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.session.run(
                self.train_op,
                feed_dict = {self.state_features:X
                            ,self.actions:actions
                            ,self.advantages:advantages}
                )
        
    def predict(self, X):
        X = np.atleast_2d(X)
        
        x_pred = self.session.run(self.lfinal, feed_dict = {self.state_features:X})
        return(x_pred)
    
    def sample_action(self, X):
        p = self.predict(X)[0]
        return (np.random.choice(len(p), p=p))
    
    def sample_eligible_action(self, X, A):
        # A is a list of integers pointing to the positions of the 
        # eligible actions
        p = self.predict(X)[0]
#        print(p)
        p_e = p[A]/sum(p[A])
        return(np.random.choice(A, p=p_e))
        
        
        
## The value model is like the critic 
## It parametrizes the value function

class ValueModel:
    def __init__(self, D):
        
        self.state_features = tf.placeholder(dtype = tf.float32, shape = (None, D), name = 'StateFeatures')
        self.state_rewards  = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = 'StateRewards')
        
        self.l1 = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid).apply(self.state_features)
        self.l2 = tf.keras.layers.Dense(units = 1, activation = tf.nn.sigmoid).apply(self.l1)
        
        
        self.mse_loss = tf.losses.mean_squared_error(self.state_rewards, self.l2)
        self.train_op = tf.train.RMSPropOptimizer(1e-4).minimize(self.mse_loss)
        
    def set_session(self, session):
        self.session = session
    
    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict={self.state_features: X, self.state_rewards: Y})
        
    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.l2, feed_dict={self.state_features: X})


## Define function for playing one round according to a given model: 

def play_general_mc(env, pmodel, vmodel, update = True, gamma = 1, Testing = False):
    
    ## Restart the game ## 
    env.reset()
    
    ## 
    states = []
    action_indeces = []
    actions  = []
    rewards = []
    total_reward = 0
    it = 0 
    
    ## Init s,r
    
    s = env.game_state ## S0
    r = env.last_score
    
    while not env.game_over:

        a_ix = pmodel.sample_action(s) ## Sample action location
#        a_e = [env.all_actions.index(j) for j in env.get_actions_set()]
#        a_ix = pmodel.sample_eligible_action(s, a_e)
        a = env.all_actions[a_ix] # point to the actual action 

        ## Update the list
        states.append(s)
        action_indeces.append(a_ix)
        actions.append(a)
        
        ## perform action 
        s, _, r = env.perform_action(a) ## perform the action and obtain the new state and the return
        rewards.append(r)
        it += 1
        
        ## Increase the reward 
        total_reward += r
        ingame_score = env.final_score
    
    ## Add the last s, a, r 
#    a_ix = pmodel.sample_action(s) ## Sample action location
#    a = env.all_actions[a_ix] # point to the actual action 
    
    ## Update the list
#    states.append(s)
    
    ## The only problem in this code is that i dont see the point sampling from 
    ## the action space after reaching the reminal state 
    
#    action_indeces.append(a_ix)
#    actions.append(a)
#    rewards.append(0) ## This is the reward after the terminal state 
    if Testing:
        return(states, actions , action_indeces, rewards)
    if update:
        ## Evaluate MC returns by iterating over the returns and the actions backwatds
        ## Init the returns with 0 
        G = 0 
        
        ## Create PH for the advantages and the returns. The advantage = G- V(S)
        advantages = []
        returns = []
        
        for st, rw in zip(reversed(states), reversed(rewards)):
            returns.append(G)
            advantages.append(G - vmodel.predict(st))
            G = rw + gamma * G
       
        returns.reverse()
        advantages.reverse()
    #    print(rewards)
        
          ## 
        adv = np.vstack(advantages)
        
        ## Update model parameters 
        pmodel.partial_fit(states, action_indeces, adv)
        vmodel.partial_fit(states, np.vstack(rewards))
        
        return(total_reward)
    
    return(ingame_score, it)





## Play the game: 
## Create instances of the value model and the policy model
pmodel = PolicyModel(D = D, K = K)
vmodel = ValueModel(D = D)

## Assign session and Init the parameters of pomodel: 
pmodel.set_session(session = tf.Session())
pmodel.session.run(tf.global_variables_initializer())
vmodel.set_session(session = tf.Session())
vmodel.session.run(tf.global_variables_initializer())
#gameinst.perform_action(('Roll', (0,0,0,0,0,0)))

train_time_reward = []
Model_scores = []
for i in range(10001):
    x = play_general_mc(env = gameinst, pmodel = pmodel, vmodel =vmodel, update = True, gamma = 1)
    train_time_reward.append(x)
    
    ## Evaluate 
    if i % 500 == 0:
        print('Finished Iteration', i, 'Evaluate reuslts')
        
        evaluation_array = []
        for i in range(100):
            
            x = play_general_mc(env = gameinst, pmodel = pmodel, vmodel =vmodel, update = False, gamma = 1)
            evaluation_array.append(x)
        Model_scores.append(evaluation_array)
        print('Evaluation finished. Coninue with the training...')



#############################
## DEBUG AND TESTING BLOCK ##
#############################
        
#pmodel_tst = PolicyModel(D = D, K = K)
#vmodel_tst = ValueModel(D = D)
#
#pmodel_tst.set_session(session = tf.Session())
#pmodel_tst.session.run(tf.global_variables_initializer())


###
#
#states_r, actions_r , action_indeces_r, rewards_r = play_general_mc(env = gameinst, pmodel = pmodel_tst, vmodel =vmodel_tst, update = False, gamma = 1, Testing = True)
#
#
#
#G1 = tf.Graph()
#
#
#with G1.as_default() as g:
#        state_features = tf.placeholder(dtype = tf.float32, shape = (None, D), name = 'StateFeatures')
#        actions = tf.placeholder(dtype = tf.int32, shape = (None,), name = 'Actions')
#        advantages = tf.placeholder(dtype = tf.float32, shape = (None,1), name = 'Advantages' )
#        
#        ## SETUP THE GRAPH:
#        l1 = tf.keras.layers.Dense(units = 40, activation = tf.nn.sigmoid).apply(state_features)
#        l2 = tf.keras.layers.Dense(units = 20, activation = tf.nn.sigmoid).apply(l1)
#        lfinal =  tf.keras.layers.Dense(units = K, activation=tf.nn.softmax).apply(l2) ## Our actual predict
#        zg = tf.one_hot(actions, K)
#        zg2 = lfinal * tf.one_hot(actions, K)
#        ## Cost function: 
#        selected_probs = tf.log(
#                            tf.reduce_sum(
#                                    lfinal * tf.one_hot(actions, K),
#                                    reduction_indices = [1]
#                                    )
#                            )
#        ## Cost function
#        cost = -tf.reduce_sum(advantages - selected_probs)
#        
#        ## TrainOp
#        train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
#        init_op = tf.global_variables_initializer()
#        
#        
#s1 = tf.Session(graph = G1)
#
#s1.run(init_op)
#
#z = s1.run(lfinal, feed_dict={state_features:states_r})
#z_act = s1.run(zg2, feed_dict={actions:np.array(action_indeces_r),state_features:states_r})
#
#
#z_act.shape