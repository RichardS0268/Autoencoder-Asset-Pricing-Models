from models.PCA import PCA
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import charas
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def model_inference_and_predict(model):
    mon_list = pd.read_pickle('data/mon_list.pkl')
    test_mons = mon_list.loc[mon_list >= model.test_period[0]]
    inference_result = []
    predict_result = []
    T_bar = tqdm(test_mons.groupby(test_mons.apply(lambda x: x//10000)), colour='red', desc=f'{model.name} Inferencing & Predicting')
    
    for g in T_bar: # rolling train
        T_bar.set_postfix({'Year': g[0]})
        model.train()
        
        for m in g[1].to_list():
            inference_result.append(model.inference(m))
            predict_result.append(model.predict(m))
        # model refit (change train period and valid period)
        model.refit()

    inference_result = pd.DataFrame(inference_result, index=test_mons, columns=charas)
    inference_result.to_csv(f'results/inference/{model.name}_inference.csv')
    
    predict_result = pd.DataFrame(predict_result, index=test_mons, columns=charas)
    predict_result.to_csv(f'results/predict/{model.name}_predict.csv')
    
    

