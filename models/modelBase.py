import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

class modelBase:
    def __init__(self, name):
        self.name = name
        self.train_idx = 0
        self.refit_cnt = 0
        
        # initial train, valid and test periods are default accroding to original paper
        self.train_period = [19570101, 19741231]
        self.valid_period = [19750101, 19861231]
        self.test_period  = [19870101, 19871231]
    
    
    def train_model(self):
        # print('trained')
        pass

    
    def calBeta(self, month):
        """
        Calculate specific month's beta. Should be specified by different models
        -> return np.array, dim = (N, K)
        """
        # return np.zeros([13000, 3])
        pass
    
        
    def calFactor(self, month):
        """
        Calculate specific month's factor. Should be specified by different models
        -> return np.array, dim = (K, 1)
        """
        # return np.zeros([3, 1])
        pass    
       
    
    def cal_delayed_Factor(self, month):
        """
        Calculate delayed month's factor, i.e. mean average of factors up to t-1. Should be specified by different models
        -> return np.array, dim = (K, 1)
        """
        pass
    
    
    def inference(self, month):       
        assert month >= self.test_period[0], f"Month error, {month} is not in test period {self.test_period}"
        
        mon_factor, mon_beta = self.calFactor(month), self.calBeta(month)
        
        assert mon_beta.shape[1] == mon_factor.shape[0], f"Dimension mismatch between mon_factor: {mon_factor.shape} and mon_beta: {mon_beta.shape}"
        
        # R_{N*1} = Beta_{N*K} @ F_{K*1}
        return mon_beta @ mon_factor
        
    
    def predict(self, month):
        assert month >= self.test_period[0] and month <= self.test_period[1], f"Month error, {month} is not in test period {self.test_period}"
        
        lag_factor, mon_beta = self.cal_delayed_Factor(month), self.calBeta(month)
        
        assert mon_beta.shape[1] == lag_factor.shape[0], f"Dimension mismatch between lag_factor: {lag_factor.shape} and mon_beta: {mon_beta.shape}"
        
        # R_{N*1} = Beta_{N*K} @ lag_F_avg{K*1}  
        return mon_beta @ lag_factor
    
    
    def refit(self):
        # self.train_period[1] += 10000 # method in original paper: increase training size by one year each time refit
        self.train_period = (pd.Series(self.train_period) + 10000).to_list() # rolling training
        self.valid_period = (pd.Series(self.valid_period) + 10000).to_list()
        self.test_period = (pd.Series(self.test_period) + 10000).to_list()
        self.refit_cnt += 1
        
        