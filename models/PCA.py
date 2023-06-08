import sys
sys.path.append('../')

import pandas as pd
import numpy as np

from utils import CHARAS_LIST
from .modelBase import modelBase


def stock_R_matrix(start_date, end_date):
    R_matrix = pd.read_pickle('data/stock_R_matrix.pkl')
    return R_matrix.T.loc[start_date: end_date].T

def portfolio_R_matrix(start_date, end_date):
    portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
    return portfolio_ret.loc[(portfolio_ret['DATE'] >= start_date) & (portfolio_ret['DATE'] <= end_date)].set_index('DATE').T



class PCA(modelBase):
    def __init__(self, K, portfolio=True):
        super(PCA, self).__init__(f'PCA_{K}')
        self.K = K
        self.portfolio = portfolio
        
        
    def __col_de_mean(self, matrix):
        return (matrix - matrix.mean()).fillna(0)
    
        
    def inference(self, month):
        if self.portfolio:
            r_matrix = self.__col_de_mean(portfolio_R_matrix(self.train_period[0], month)).astype(np.float32)
        else:
            r_matrix = self.__col_de_mean(stock_R_matrix(self.train_period[0], month))   
        u, sigma, vt = np.linalg.svd(r_matrix)
        # B_{N*K}
        B = u[:, :self.K]
        # F_{K*1}
        F = np.diag(sigma[:self.K]) @ vt[:self.K, -1]
        return B @ F
        
            
    def predict(self, month):
        if self.portfolio:
            r_matrix = self.__col_de_mean(portfolio_R_matrix(self.train_period[0], month)).astype(np.float32)
        else:
            r_matrix = self.__col_de_mean(stock_R_matrix(self.train_period[0], month))
        u, sigma, vt = np.linalg.svd(r_matrix)
        # B_{N*K}
        B = u[:, :self.K]
        # F_{K*1}
        lag_F = np.diag(sigma[:self.K]) @ vt[:self.K, :-1]
        return B @ np.mean(lag_F, axis=1)
    
    