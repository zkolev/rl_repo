# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:53:51 2019

@author: zkolev


This is an implementation of the Proximal Policy Optimization loss for discrete action space will be used in actor-critic framework 
It is used to optimize model that simultaneously trains the policy and the value head with added entropy regularization 
Reference paper: https://arxiv.org/abs/1707.06347

"""

## Contains training ops

import tensorflow as tf
import numpy as np 


def PPO_Loss(model
				,data
				, returns
				, old_prediction
				, actions
				, total_actions
				, epsilon = 0.2
				, entropy_reg = 0.01
				, value_reg = 0.5
				):
    """
    
    model: tensorflow (or keras) odel that takes "data" as input and yields action logits
    data: list of tensors / arrays  
    returns: The total return  
    old_predictions: needed for the ppo objective softmax(logits) with size [batch_size, total_actions]
    actions = vector of integers with max_value = (total_actions -1)
    total_actions = the dim of the matrix
    epsilon = cliping constant 
    entropy_reg =  categorical entropy reg constant
    value_reg = value loss reg constant 
    
    """
    
	 ## 1-hot encode the action vector
    y_true = tf.one_hot(actions, depth = total_actions)
    
    ## The model is with two output heads
    ## 1 yields the logits of pi(a|s) and the other V(s)
    
    y_pred , y_value = model(data)
    
    ## Transform logits to probabilities
    y_pred_prob = tf.nn.softmax(y_pred)
    
    ## Calculate the advantages 
    ## should yield tensor with dim (n_obs x 1 )
    advantages = returns - y_value
    
    
    ## The logits will be used for entropy reg
    prob = tf.math.reduce_sum(y_true * y_pred_prob, axis = 1)
    old_prob = tf.math.reduce_sum(y_true * old_prediction, axis = 1)
    r_theta = prob/(old_prob + 1e-10)
    clipped = tf.clip_by_value(r_theta, 1 - epsilon, 1 + epsilon)
    ppo_loss = tf.math.reduce_mean(tf.math.reduce_min(tf.concat([r_theta * advantages, clipped * advantages], axis= 1), axis = 1))
    
    ## Calculate entropy: 
    a0 = y_pred - tf.math.reduce_max(y_pred, axis = 1, keepdims a= True)
    ea0 = tf.math.exp(a0)
    z0 = tf.reduce_sum(ea0, 1 , keepdims = True)
    p0 = ea0/z0
    entropy = tf.reduce_mean(tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1))
    
    ## Calculate value loss
    ## Value loss is simple MSE 
    mse_vf = tf.reduce_mean(tf.losses.mse(returns, y_value))
    
    policy_loss = ppo_loss - entropy * entropy_reg + mse_vf*value_reg
    
    return(policy_loss)