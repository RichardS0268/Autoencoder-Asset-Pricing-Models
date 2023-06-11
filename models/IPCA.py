import pandas as pd
import numpy as np

import sys
sys.path.append('../')

from utils import *
from .modelBase import modelBase


class IPCA(modelBase):
    def __init__(self, K, omit_char=[]):
        super(IPCA, self).__init__(f'IPCA_{K}')
        self.K = K
        self.omit_char = omit_char
        np.random.seed(10)
        self.gamma = np.random.random([94, self.K]) # P = 94, we have total 94 characteristics 
        self.valid_error = []
        self.__prepare_data()
        

    def __prepare_data(self):
        self.portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
        self.p_charas = pd.read_pickle('data/p_charas.pkl')
        self.mon_list = pd.read_pickle('data/mon_list.pkl')
    
        
    def __valid(self):
        MSE_set = []
        for mon in self.mon_list[(self.mon_list >= self.valid_period[0]) & (self.mon_list <= self.valid_period[1])]:
            Z = self.p_charas.loc[self.p_charas.DATE == mon][CHARAS_LIST].values # N * P
            y = self.portfolio_ret.loc[self.portfolio_ret.DATE == mon][CHARAS_LIST].values.T # N * 1
            beta = Z @ self.gamma # N * K
            f_hat = np.array(np.matrix(beta.T @ beta).I @ beta.T @ y) # K * 1
            residual = y - beta @ f_hat
            MSE = np.sum(residual**2)
            MSE_set.append(MSE)
            
        valid_error = sum(MSE_set)
        self.valid_error.append(valid_error)
        
        return valid_error
    
        
    def __gamma_iter(self, gamma_old):
        numer = np.zeros((94*self.K, 1))
        denom = np.zeros((94*self.K, 94*self.K))
        for mon in self.mon_list[(self.mon_list >= self.train_period[0]) & (self.mon_list <= self.train_period[1])]:
            Z = self.p_charas.loc[self.p_charas.DATE == mon][CHARAS_LIST].values # N * P
            y = self.portfolio_ret.loc[self.portfolio_ret.DATE == mon][CHARAS_LIST].values.T # N * 1
            beta = Z @ gamma_old # N * K
            f_hat = np.array(np.matrix(beta.T @ beta).I @ beta.T @ y) # K * 1
            numer += (np.kron(f_hat, Z.T) @ y)
            denom += (np.kron(f_hat, Z.T) @ np.kron(f_hat.T, Z))
            
        gamma_new = (np.linalg.pinv(denom) @ numer).reshape(self.K, 94)
        gamma_new = gamma_new.T    

        return gamma_new
    

    def train_model(self):
        update_cnt = 0
        min_valid_err = np.Inf
        best_gamma = np.zeros((94, self.K)) 
        while update_cnt < 5:
            self.gamma = self.__gamma_iter(self.gamma)
            valid_error = self.__valid()
            if valid_error < min_valid_err:
                min_valid_err = valid_error
                best_gamma = self.gamma
                update_cnt = 0
            else:
                update_cnt += 1
        
        self.gamma = best_gamma
        
    
    def inference(self, month):
        if not len(self.omit_char):        
            Z = self.p_charas.loc[self.p_charas.DATE == month][CHARAS_LIST].values # N * P
            y = self.portfolio_ret.loc[self.portfolio_ret.DATE == month][CHARAS_LIST].values.T # N * 1
            beta = Z @ self.gamma # N * K
            f_hat = np.array(np.matrix(beta.T @ beta).I @ beta.T @ y) # K * 1
            return (beta @ f_hat).flatten() # N, 1
        else:
            inference_R = []
            Z = self.p_charas.loc[self.p_charas.DATE == month][CHARAS_LIST].copy(deep=False)
            y = self.portfolio_ret.loc[self.portfolio_ret.DATE == month][CHARAS_LIST].copy(deep=False)
            
            for char in self.omit_char:
                Z_input = Z.copy(deep=False)
                y_input = y.copy(deep=False)
                Z_input[[char]] = Z_input[[char]] * 0.0
                y_input[[char]] = y_input[[char]] * 0.0
                Z_input = Z_input.values
                y_input = y_input.values.T
                beta = Z_input @ self.gamma
                f_hat = np.array(np.matrix(beta.T @ beta).I @ beta.T @ y_input) # K * 1
                inference_R.append((beta @ f_hat).flatten()) # m * N

            Z_input = Z.values
            y_input = y.values.T
            beta = Z_input @ self.gamma
            f_hat = np.array(np.matrix(beta.T @ beta).I @ beta.T @ y_input) # K * 1
            inference_R.append((beta @ f_hat).flatten()) # m * N
            
            return np.array(inference_R).T # N * m
    
    
    def predict(self, month):
        if self.refit_cnt == 0:
            return self.inference(month)
        
        lag_f_hat = []
        for mon in self.mon_list[(self.mon_list >= 19870101) & (self.mon_list < month)]:
            Z = self.p_charas.loc[self.p_charas.DATE == mon][CHARAS_LIST].values # N * P
            y = self.portfolio_ret.loc[self.portfolio_ret.DATE == mon][CHARAS_LIST].values.T # N * 1
            beta = Z @ self.gamma # N * K
            f_hat = np.array(np.matrix(beta.T @ beta).I @ beta.T @ y) # K * 1
            lag_f_hat.append(f_hat)
            
        Z = self.p_charas.loc[self.p_charas.DATE == month][CHARAS_LIST].values # N * P
        y = self.portfolio_ret.loc[self.portfolio_ret.DATE == month][CHARAS_LIST].values.T # N * 1
        beta = Z @ self.gamma # N * K
        
        # return average of prevailing sample hat{f} (from 198701) up to t-1
        avg_lag_f = np.mean(lag_f_hat, axis=0)
        return beta @ avg_lag_f