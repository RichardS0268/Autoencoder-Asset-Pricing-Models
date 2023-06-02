import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

class modelBase:
    def __init__(self, name):
        self.name = name
        self.train_idx = 0
        
        # initial train, valid and test periods are default accroding to original paper
        self.train_period = [19570101, 19741231] 
        self.valid_period = [19750101, 19861231]
        self.test_period  = [19870101, 19871231]
        
        # beta_fn and factor_fn are functions to generate beta and factor. They can be float (for FF, PCA), linear function (for IPCA) or NN (for CA, CAA)
        self.beta_fn = None
        self.factor_fn = None
    
    def train(self):
        # print('trained')
        pass

    
    def calBeta(self, month):
        '''
        Calculate specific month's beta. Should be specified by different models
        -> return np.array, dim = (N, K)
        '''
        return np.zeros([3, 1])
        pass
    
        
    def calFactor(self, month):
        '''
        Calculate specific month's factor. Should be specified by different models
        -> return np.array, dim = (K, 1)
        '''
        return np.zeros([1300, 3])
        pass    
       
    
    def inference(self, month):        
        assert month >= self.test_period[0], f"Month error, {month} is not in test period {self.test_period}"
        
        mon_factor, mon_beta = self.calFactor(month), self.calBeta(month)
        
        assert mon_factor.shape[1] == mon_beta.shape[0], f"Dimension mismatch between mon_factor: {mon_factor.shape} and mon_beta: {mon_beta.shape}"
        
        # R_{N*1} = Beta_{N*K} @ factor_{K*1}
        return mon_factor @ mon_beta 
        
    
    def predict(self, month):
        assert month >= self.test_period[0] and month <= self.test_period[1], f"Month error, {month} is not in test period {self.test_period}"
        
        mon_beta = self.calBeta(month)
        
        mon_start = datetime.datetime.strptime(str(self.valid_period[0]),'%Y%m%d')
        mon_end = datetime.datetime.strptime(str(month),'%Y%m%d')
        
        lag_factor_list = []
        while mon_start < mon_end - relativedelta(months=1):
            # lag_factor_{K*1}
            lag_factor = self.calFactor(int(str(mon_start).split(' ')[0].replace('-', '')))
            lag_factor_list.append(lag_factor)
            mon_start += relativedelta(months=1)
        
        # lag_factor_avg_{N*K}
        lag_factor_avg = np.array(lag_factor_list).mean(axis=0)
        
        assert lag_factor_avg.shape[1] == mon_beta.shape[0], f"Dimension mismatch between lag_factor_avg: {lag_factor_avg.shape} and mon_beta: {mon_beta.shape}"
        
        # R_{N*1} = lag_F_{N*K} @ Beta_{K*1}
        return lag_factor_avg @ mon_beta
    
    
    def refit(self):
        self.train_period[1] += 10000
        self.valid_period = (pd.Series(self.valid_period) + 10000).to_list()
        self.test_period = (pd.Series(self.test_period) + 10000).to_list()

        self.train()
        
        self.train_idx += 1
        print(f'Model has been refitted [{self.train_idx}]')
        