import sys
sys.path.append('../')

from ipca import InstrumentedPCA

import pandas as pd
import numpy as np

from utils import charas
from .modelBase import modelBase



class IPCA(modelBase):
    def __init__(self, K, portfolio=True):
        super(IPCA, self).__init__(f'IPCA_{K}')
        self.K = K
        self.portfolio = portfolio
        self.__prepare_data()

    def __prepare_data(self):
        self.p_charas = pd.read_pickle('data/p_charas.pkl')
        portfolio_ret=  pd.read_pickle('data/portfolio_ret.pkl')
        self.p_charas['p_ret'] = np.zeros(self.p_charas.shape[0])
        self.train_p_charas = self.p_charas.loc[self.p_charas.DATE <= self.test_period[1]].copy(deep=False).reset_index().set_index(['index', 'DATE']).sort_index()
        for chr in charas:
            self.train_p_charas.loc[f'p_{chr}', 'p_ret'] = portfolio_ret.loc[portfolio_ret.DATE <= self.test_period[1]][chr].values

        
    def train_model(self):
        y = self.train_p_charas['p_ret']
        X = self.train_p_charas.drop('p_ret', axis=1)

        regr = InstrumentedPCA(n_factors=1, intercept=False)
        regr = regr.fit(X=X, y=y)
        self.Gamma, self.Factors = regr.get_factors(label_ind=True)
        
    
    def calBeta(self, month):
        return self.p_charas.loc[self.p_charas.DATE == month][charas].values @ self.Gamma.values # (N * P) @ (P * K) -> (N, K)
    
    def calFactor(self, month):
        return self.Factors.values[:, -1] # K * 1

    def cal_delayed_Factor(self, month):
        return np.mean(self.Factors.values[:, :-1], axis=1)
    