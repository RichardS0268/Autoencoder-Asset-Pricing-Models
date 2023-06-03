from models.PCA import PCA
from models.FF import FF

import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import charas
import matplotlib.pyplot as plt
import gc

from models.CA import CA0, CA1, CA2

import warnings
warnings.filterwarnings('ignore')

def model_inference_and_predict_CA(model):
    mon_list = pd.read_pickle('data/mon_list.pkl')
    test_mons = mon_list.loc[mon_list >= model.test_period[0]]
    inference_result = pd.DataFrame()
    predict_result = pd.DataFrame()
    T_bar = tqdm(test_mons.groupby(test_mons.apply(lambda x: x//10000)), colour='red', desc=f'{model.name} Inferencing & Predicting')
    
    stock_index = pd.Series(dtype=np.int64)
    for g in T_bar: # rolling train
        T_bar.set_postfix({'Year': g[0]})
        model.train_model()
        
        for m in g[1].to_list():
            m_stock_index, _, _, _ = model._get_item(m)
            stock_index = pd.concat([stock_index, pd.Series(m_stock_index)]).drop_duplicates().astype(int)
            inference_R = model.inference(m) # return (N, 1)
            predict_R = model.inference(m) # reutrn (N, 1)

            # move inference_R and predict_R to cpu
            inference_R = inference_R.cpu().detach().numpy()
            predict_R = predict_R.cpu().detach().numpy()

            inference_R = pd.DataFrame(inference_R, index=m_stock_index, columns=[m])
            predict_R = pd.DataFrame(predict_R, index=m_stock_index, columns=[m])

            
            inference_result = pd.concat([inference_result.reset_index(drop=True), inference_R.reset_index(drop=True)], axis=1) # (N, T)
            predict_result = pd.concat([predict_result.reset_index(drop=True), predict_R.reset_index(drop=True)], axis=1) # (N, T)

            # DEBUG:
            # save inference_R and predict_R to csv
            # inference_result.to_csv(f'temp/{model.name}_inference_stock_{m}.csv')
            # predict_result.to_csv(f'temp/{model.name}_predict_stock_{m}.csv')
            
        # model refit (change train period and valid period)
        model.refit()

    inference_result = pd.DataFrame(inference_result.values.T, index=test_mons, columns=charas)
    inference_result.to_csv(f'results/inference/{model.name}_inference.csv')
    
    predict_result = pd.DataFrame(predict_result.values.T, index=test_mons, columns=charas)
    predict_result.to_csv(f'results/predict/{model.name}_predict.csv')
    return inference_result, predict_result

OOS_start = 19870101
OOS_end = 20161231

def calculate_R2(model, type, portfolio = True):
    if portfolio:
        portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
    else:
        raise Exception('Unrealized Function')
    oos_ret = portfolio_ret.loc[(portfolio_ret['DATE'] >= OOS_start) & (portfolio_ret['DATE'] <= OOS_end)]
    
    output_path = f'results/{type}/{model}_{type}.csv'
    model_output = pd.read_csv(output_path)
    
    residual_square = (oos_ret.set_index('DATE') - model_output.set_index('DATE'))**2
    residual_square = (1 - (residual_square == np.inf) * 1.0) * residual_square # drop Inf outliers
    
    total_square = oos_ret.set_index('DATE')**2
    total_square = (1 - (total_square == np.inf) * 1.0) * total_square # drop Inf outliers
    
    return 1 - np.sum(residual_square.values)/np.sum(total_square.values)

def alpha_plot(model, type, portfolio = True):
    if not portfolio:
        raise Exception('Unrealized Function')
    
    portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
    oos_result = portfolio_ret.loc[(portfolio_ret['DATE'] >= OOS_start) & (portfolio_ret['DATE'] <= OOS_end)].set_index('DATE')
    
    output_path = f'results/{type}/{model}_{type}.csv'
    inference_result = pd.read_csv(output_path)
    inference_result = inference_result.set_index('DATE')
    
    
    pricing_error_analysis = []
    for col in charas:
        raw_return = oos_result[col].mean()
        error = oos_result[col] - inference_result[col]
        alpha = error.mean()
        t_stat = abs(error.mean()/error.std()) * np.sqrt(oos_result.shape[0])
        pricing_error_analysis.append([raw_return, alpha, t_stat])

    pricing_error_analysis = pd.DataFrame(pricing_error_analysis, columns = ['raw ret', 'alpha', 't_stat'], index=charas)
    
    lower_point = min(np.min(pricing_error_analysis['raw ret']), np.min(pricing_error_analysis['alpha'])) * 1.15
    upper_point = max(np.max(pricing_error_analysis['raw ret']), np.max(pricing_error_analysis['alpha'])) * 1.15

    significant_mask = pricing_error_analysis['t_stat'] > 3

    plt.scatter(pricing_error_analysis.loc[significant_mask]['raw ret'], pricing_error_analysis.loc[significant_mask]['alpha'], marker='^', color='r', alpha=0.6, label=f'#Alphas(t>3.0)={np.sum(significant_mask*1.0)}')
    plt.scatter(pricing_error_analysis.loc[~significant_mask]['raw ret'], pricing_error_analysis.loc[~significant_mask]['alpha'], marker='o', color='b', alpha=0.6, label=f'#Alphas(t<3.0)={94-np.sum(significant_mask*1.0)}')
    plt.plot(np.linspace(lower_point, upper_point, 10), np.linspace(lower_point, upper_point, 10), color='black')

    plt.ylabel('Alpha')
    plt.xlabel('Raw Return')
    plt.legend()

    plt.title(model)
    plt.savefig(f'results/{type}/{model}_{type}_alpha_plot.png')

def main():
    # CA0
    CA0_list = []
    for k in range(6):
        gc.collect()
        CA0_list.append(CA0(k+1).to('cuda'))
        model_inference_and_predict_CA(CA0_list[k])

    # CA1
    CA1_list = []
    for k in range(6):
        gc.collect()
        CA1_list.append(CA1(k+1).to('cuda'))
        model_inference_and_predict_CA(CA1_list[k])

    # CA2
    CA2_list = []
    for k in range(6):
        gc.collect()
        CA2_list.append(CA2(k+1).to('cuda'))
        model_inference_and_predict_CA(CA2_list[k])

    # calc R2
    # R2 is a dataframe with index = layer
    R2 = pd.DataFrame()
    for l in range(3):
        for k in range(6):
            R2[f'CA{l}_k{k+1}'] = calculate_R2(f'CA{l}_k{k+1}', 'inference')
            alpha_plot(f'CA{l}_k{k+1}', 'inference')

    R2.to_csv('results/R2.csv')

if __name__ == '__main__':
    main()
    