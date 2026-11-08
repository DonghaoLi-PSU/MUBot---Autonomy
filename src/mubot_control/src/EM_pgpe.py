# This Python file uses the following encoding: utf-8

# if__name__ == "__main__":
#     pass
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 2021
Update on
@author: donghao Li
"""

import numpy as np
# import rospy

class EMPGPE():
    def __init__(self, n_parmas,n_rollout,act_num,select):
        self.n_parmas   = n_parmas            # number of parameter
        self.n_rollout  = n_rollout          # number of rollout
        self.act_num    = act_num              # number of actuator
        self.mu         = np.zeros((n_parmas))      # mean of policy parameter
        self.sigma      = np.zeros((n_parmas))   # standard deviation of policy parameter
        self.select     = select
        self.sig_range  = np.zeros(n_parmas)
        # signma constraint
        self.sig_range[0] = 1E-2            #first amplitude
        for i in range (act_num-1):
            self.sig_range[i*2+1] = 1E-2    #amplitude
            self.sig_range[i*2+2] = 1E-6    #phase
        self.sig_range[-1] = 1E-6           #frequency

    def sample_params(self, mu, sigma):
        s_param = np.zeros([self.n_rollout,self.n_parmas])
        for i in range(self.n_parmas):
            s_param[:,i] = np.random.normal(mu[i],sigma[i],self.n_rollout)
        return s_param

    def learn(self, thetas, reward, pre_mu, pre_sigma):
#        reward = np.array()
        sorted_index = reward.argsort()[::-1][:self.select]
        s_reward = reward[sorted_index]+0.1
        s_thetas = thetas[sorted_index]
        ###Update mu and sigma
        new_mu = sum(s_reward[:,None]*s_thetas)/sum(s_reward)
        for i in range(self.act_num-1):
            new_mu[2*i+2] = new_mu[2*i+2]%1
        new_sigma = np.sqrt( sum( s_reward[:,None]*np.square(s_thetas-new_mu[None,:]) )/sum(s_reward) )

        ###Constrain
        new_sigma = np.clip(new_sigma,0.8*pre_sigma,None)
        new_sigma = np.clip(new_sigma,self.sig_range,None)

        ###Transport
        delta_mu = new_mu-pre_mu
        self.mu = new_mu
        self.sigma = new_sigma
        return new_mu, new_sigma, delta_mu



