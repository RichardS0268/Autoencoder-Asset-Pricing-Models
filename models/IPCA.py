import sys
sys.path.append('../')

from ipca import InstrumentedPCA
import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

from utils import CHARAS_LIST, HiddenPrints
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
        with HiddenPrints():
            y = self.train_p_charas['p_ret']
            X = self.train_p_charas.drop('p_ret', axis=1)

            self.regr = InstrumentedPCA(n_factors=self.K, intercept=False)
            self.regr = self.regr.fit(X=X, y=y)
            self.Gamma, self.Factors = self.regr.get_factors(label_ind=False)
        
    
    def inference(self, month):
        X_pred = self.p_charas.drop('p_ret', axis=1).loc[self.p_charas.DATE == month].copy(deep=False).reset_index().set_index(['index', 'DATE']).sort_index()
        return self.regr.predict(X_pred, mean_factor=True) # (N, 1)
    
    def predict(self, month):
        lag_X = self.p_charas.drop('p_ret', axis=1).loc[self.p_charas.DATE < month].copy(deep=False).reset_index().groupby('index').mean()
        lag_X.DATE = self.p_charas.loc[self.p_charas.DATE < month].DATE.drop_duplicates()[-1]
        lag_X = lag_X.reset_index().set_index(['index', 'DATE']).sort_index()
        return self.regr.predict(lag_X, mean_factor=True) # (N, 1)
    