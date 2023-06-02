import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from tqdm import tqdm

import os
import zipfile
from joblib import delayed, Parallel
from itertools import product
from utils import charas

import warnings
warnings.filterwarnings('ignore')

if 'data.zip' not in os.listdir():
    os.system('wget https://cloud.tsinghua.edu.cn/f/07d6a0223d054247af26/?dl=1 -O data.zip')

if 'data' not in os.listdir():
    os.mkdir('data')
    os.system('wget https://cloud.tsinghua.edu.cn/f/179082ecf0f147a4840c/?dl=1 -O portfolio_ret.pkl')
    os.system('wget https://cloud.tsinghua.edu.cn/f/b93c6ae7e2014d3a951e/?dl=1 -O ff5.csv')
    os.system('wget https://cloud.tsinghua.edu.cn/f/5f077be9eda0428ab7e5/?dl=1 -O UMD.csv')
    
    os.system('mv portfolio_ret.pkl data')
    os.system('mv ff5.csv data')
    os.system('mv UMD.csv data')
    
    
with zipfile.ZipFile('data.zip', 'r') as z:    
    with z.open('data/month_ret.pkl') as f:
        print('Reading month_ret.pkl', end=' ')
        mon_ret = pd.read_pickle(f)    
        mon_ret.to_pickle('data/month_ret.pkl')
        print('Done!')
        
    with z.open('data/datashare.pkl') as f:
        print('Reading datashare.pkl', end=' ')
        datashare = pd.read_pickle(f)
        datashare['DATE'].drop_duplicates().reset_index(drop=True).to_pickle('data/mon_list.pkl')
        # datashare.to_pickle('data/datashare.pkl')
        print('Done!')



def pre_process(date):
    cross_slice = datashare.loc[datashare.DATE == date].copy(deep=False)
    
    omitted_mask = 1.0 * np.isnan(cross_slice.loc[cross_slice['DATE'] == date])
    # fill nan values with each factors median
    cross_slice.loc[cross_slice.DATE == date] = cross_slice.fillna(0) + omitted_mask * cross_slice.median()
    # if all stocks' factor is nan, fill by zero
    cross_slice.loc[cross_slice.DATE == date] = cross_slice.fillna(0)
    # rank-normalize all characteristics into the interval [-1, 1]
    cross_slice.loc[cross_slice.DATE == date, charas] = (((cross_slice - cross_slice.min()) / (cross_slice.max() - cross_slice.min()))[charas].fillna(0.5) * 2 - 1).astype(np.float16)
    
    return cross_slice


def cal_portfolio_ret(it, df):
    d, f = it[0], it[1]
    # long portfolio, qunatile 0.0~0.1; short portfolio, qunatile 0.9~1.0
    long_portfolio = df.loc[df.DATE == d][['permno', f]].sort_values(by=f, ascending=False)[:df.loc[df.DATE == d].shape[0]//10]['permno'].to_list()
    short_portfolio = df.loc[df.DATE == d][['permno', f]].sort_values(by=f, ascending=False)[-df.loc[df.DATE == d].shape[0]//10:]['permno'].to_list()
    # long-short portfolio return
    long_ret = mon_ret.loc[mon_ret.date == d].drop_duplicates('permno').set_index('permno').reindex(long_portfolio)['ret-rf'].dropna().mean()
    short_ret = mon_ret.loc[mon_ret.date == d].drop_duplicates('permno').set_index('permno').reindex(short_portfolio)['ret-rf'].dropna().mean()
    chara_ret = 0.5*(long_ret - short_ret)
    
    return chara_ret



if __name__ == '__main__':
    # pre-process share data
    processed_df = Parallel(n_jobs=-1)(delayed(pre_process)(d) for d in tqdm(datashare.DATE.drop_duplicates().to_list(), colour='green', desc='Processing'))
    processed_df = pd.concat(processed_df)
    processed_df[['permno', 'DATE']] = processed_df[['permno', 'DATE']].astype(int)

    ##TODO: calculate portfolio returns (or download preprocessed data)
    # iter_list = list(product(datashare.DATE.drop_duplicates(), charas))
    # portfolio_rets = Parallel(n_jobs=-1)(delayed(cal_portfolio_ret)(it, df=processed_df) for it in tqdm(iter_list, colour='green', desc='Calculating'))
    # portfolio_rets = pd.DataFrame(np.array(portfolio_rets).reshape(-1, 94), index=datashare.DATE.drop_duplicates(), columns=charas).reset_index()
    # portfolio_rets[charas] = portfolio_rets[charas].astype(np.float16)
    
    mon_list = []
    permno_index = pd.Series(dtype='float64')
    R_matrix = pd.DataFrame()

    for g in tqdm(mon_ret.groupby('date'), colour='blue', desc='Generating R Matrix'):
        mon_list.append(g[0])
        mon_r = g[1].drop_duplicates('permno')
        permno_index = pd.concat([permno_index, mon_r['permno']]).drop_duplicates()
        
        R_matrix = pd.concat([R_matrix.reindex(permno_index), mon_r.set_index('permno').reindex(permno_index)['ret-rf']], axis=1)
        R_matrix.columns = mon_list
        
    
    processed_df.to_pickle('data/datashare_re.pkl')
    # portfolio_rets.to_pickle('data/portfolio_rets.pkl')
    R_matrix.to_pickle('data/stock_R_matrix.pkl')