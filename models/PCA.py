import sys
sys.path.append('../')

import pandas as pd
import numpy as np

from utils import *
from .modelBase import modelBase


def stock_R_matrix(start_date, end_date):
    R_matrix = pd.read_pickle('data/stock_R_matrix.pkl')
    return R_matrix.T.loc[start_date: end_date].T

def portfolio_R_matrix(start_date, end_date):
    portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
    return portfolio_ret.loc[(portfolio_ret['DATE'] >= start_date) & (portfolio_ret['DATE'] <= end_date)].set_index('DATE').T



class PCA(modelBase):
    def __init__(self, K, omit_char=[]):
        super(PCA, self).__init__(f'PCA_{K}')
        self.K = K
        self.omit_char = omit_char
        self.portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
    
    def train_model(self):
        pr = self.portfolio_ret.loc[(self.portfolio_ret.DATE >= self.train_period[0]) & (self.portfolio_ret.DATE <= self.train_period[1])][CHARAS_LIST].values
        pr = pr - np.mean(pr, axis=0) # col demean
        ret_cov_matrix = np.zeros((pr.shape[1], pr.shape[1]))
        
        for i in range(pr.shape[0]): # Sum of y^t @ y^t.T
            ret_cov_matrix += (pr[i, :].reshape(-1, 1) @ pr[i, :].reshape(-1, 1).T)
        ret_cov_matrix = ret_cov_matrix/(pr.shape[0]*pr.shape[1]) # N * N

        eigVal, eigVec = np.linalg.eig(ret_cov_matrix)
        sorted_indices = np.argsort(eigVal)
        self.beta = eigVec[:,sorted_indices[:-self.K-1:-1]] # Beta: N * K
    
    
    def calBeta(self, month):
        return np.real(self.beta)
    
        
    def calFactor(self, month):
        tr = self.portfolio_ret.loc[self.portfolio_ret.DATE <= month].iloc[-1][CHARAS_LIST].values
        tr = tr - np.mean(tr, axis=0) # col demean
        # print(tr)
        factor = np.array((np.matrix(self.beta.T @ self.beta).I @ self.beta.T) @ tr.T).T # K * 1
        return np.real(factor.flatten())
    

    def cal_delayed_Factor(self, month):
        if self.refit_cnt == 0:
            return self.calFactor(month)
        
        tr = self.portfolio_ret.loc[(self.portfolio_ret.DATE >= 19870101) & (self.portfolio_ret.DATE < month)][CHARAS_LIST].values
        tr = tr - np.mean(tr, axis=0) # col demean
        
        # return average of prevailing sample hat{f} (from 198701) up to t-1
        factors = []
        for i in range(tr.shape[0]):
            factors.append(np.array(np.matrix(self.beta.T @ self.beta).I @ self.beta.T @ tr[i, :]))

        factors = np.array(factors).squeeze(1).T # K * T
        avg_delay_f = np.mean(factors, axis=1).reshape(-1, 1) # K * 1
        
        return np.real(avg_delay_f.flatten())
    