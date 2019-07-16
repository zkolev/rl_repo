# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:36:24 2019

@author: zkolev
"""

import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

from GeneralMini import GeneralGame


## Define feature transformer class:

class FeatureTransformer:
    def __init__(self, env):
        
        observation_examples = env.get_state_space()
        scaler = StandardScaler()
        scaler.fit(observation_examples)
        
        feature_set = FeatureUnion([
                 ('rbf1', RBFSampler(gamma = 5.0, n_components=250))
                ,('rbf2', RBFSampler(gamma = 2.0, n_components=250))
                ,('rbf3', RBFSampler(gamma = 1.0, n_components=250))
                ,('rbf4', RBFSampler(gamma = 0.5, n_components=250))
                ])
        
        feature_set.fit(scaler.transform(observation_examples))
        
        self.scaler = scaler
        self.feature_set = feature_set 
        
    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return(self.feature_set.transform(scaled))
    
    
    
## Utility function
def max_dict(d):
    ## Init phs:
    max_key = None
    max_val = float('-inf')
    for k,v in d.items():
        if v > max_val:
            max_key = k
            max_val = v
    
    return((max_key, max_val))

## Define the model class

class Model:
    def __init__(self, env, feat_transf):
        self.env = env
        self.models = {}
        self.actions = []
        self.feat_transf=  feat_transf
        
        for a in env.get_actions_set():
            model = SGDRegressor()
            ## Init the model with 1 step of gradient descent 
            model.partial_fit(feat_transf.transform([env.game_state]), [0])
            self.models[a] = model
            self.actions.append(a)
    
        
    def update_models(self, a):
        model = SGDRegressor()
        model.partial_fit(self.feat_transf.transform([self.env.game_state]), [0])
        self.models[a] = model
        
    def predict(self, s):
        X = self.feat_transf.stransform([s])
        assert(len(X.shape) == 2)
        return(np.array(m.predict(X)[0] for m in self.models))
        
    def update(self, s ,a ,G):
        X = self.feat_transf.stransform([s])
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X, [G])


def play_one(model, eps, gamma):
    
    ## Init the game isntance 
    game_instance = env.reset()
    
    while not game_instance.game_over:
        
        ## Check the 
    


        
        
### TEST
X = GeneralGame()
Y = FeatureTransformer(env = X)


tmp = Y.transform([X.game_state])
tmp.shape