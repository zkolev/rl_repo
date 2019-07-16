# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:54:29 2019

@author: zkolev
"""

import tensorflow as tf 
import numpy as np
from GeneralMini import GeneralGame


##


class DQN:
    def __init__(self, D, K):
        
        ## D = Number fo features of the environment
        ## K = Number of actions 
        
        ## Construct the network  ##
        
        ## Input:
        self.X = tf.placeholder(tf.float32, shape = (None,D), name = 'X')
        self.G = tf.placeholder(tf.float32, shape = (None, ), name = 'G')
        self.actions = tf.placeholder(tf.int32, shape = (None, ), name = 'actions')
        
        
        ## Graph: 
        
        self.L1 = tf.keras.layers.Dense(20, activation = tf.nn.leaky_relu, name = 'layerOne').apply(self.X)
        self.L2 = tf.keras.layers.Dense(10, activation = tf.nn.leaky_relu, name = 'layerTwo').apply(self.L1)
        
        # Final layer: 
        
        self.FinalLayer = tf.keras.layers.Dense(K, activation = None, name = 'finalLayer')
        
        ## The following scripts takes two matrices
        ## 20 X  #actions X K 
    
    def set_session(self, session):
        self.session = session
    
    def test_fw_op(self, X):
        x = self.session.run(self.L2, feed_dict = {self.X:X})
        return(x)
        
        
### 
        

X_arr = np.array([[1,2,3,4,5], [5,3,1,5,1], [1,3,4,5,1]])
MyNet = DQN(D = 5, K = 10)
MyNet.set_session(session = tf.Session())
MyNet.session.run(tf.global_variables_initializer())
x_tst = MyNet.test_fw_op(X = X_arr)

x_tst.shape


with tf.Session() as s:
    print(s.run(tf.square(2)))