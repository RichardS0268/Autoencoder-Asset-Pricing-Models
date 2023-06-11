import os
import pandas as pd
import numpy as np
import collections
from .modelBase import modelBase
from utils import CHARAS_LIST

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


MAX_EPOCH = 200

class CA_base(nn.Module, modelBase):
    def __init__(self, name, omit_char=[], device='cuda'):
        nn.Module.__init__(self)
        modelBase.__init__(self, name)
        self.beta_nn = None
        self.factor_nn = None
        self.optimizer = None
        self.criterion = None
        self.omit_char = omit_char
        
        self.factor_nn_pred = []
        
        self.device = device

        self.datashare_chara = pd.read_pickle('./data/datashare_re.pkl').astype(np.float64)
        self.p_charas = pd.read_pickle('./data/p_charas.pkl').astype(np.float64).reset_index()
        self.portfolio_ret=  pd.read_pickle('./data/portfolio_ret.pkl').astype(np.float64)
        self.mon_ret = pd.read_pickle('./data/month_ret.pkl').astype(np.float64)

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
    
    
    def debug(self, month):
        beta_nn_input = self.p_charas.loc[self.p_charas['DATE'] == month][CHARAS_LIST]
        # beta_nn_input = self.datashare_chara.loc[self.datashare_chara['DATE'] == month].set_index('permno')[charas]
        print(beta_nn_input)


    def _get_item(self, month):
        if month not in self.p_charas['DATE'].values:
            # find the closest month in p_charas to month
            month = self.p_charas['DATE'].values[np.argmin(np.abs(self.p_charas['DATE'].values - month))]
            
        beta_nn_input = self.p_charas.loc[self.p_charas['DATE'] == month][CHARAS_LIST] # (94, 94)
        labels = self.portfolio_ret.loc[self.portfolio_ret['DATE'] == month][CHARAS_LIST].T.values # (94, 1)
        beta_nn_input['ret-rf'] = labels
        align_df = beta_nn_input.copy(deep=False).dropna()
            
        factor_nn_input = self.portfolio_ret.loc[self.portfolio_ret['DATE'] == month][CHARAS_LIST]
         
        # exit(0) if there is any nan in align_df
        if align_df.isnull().values.any():
            assert False, f'There is nan in align_df of : {month}'
        # return stock index (L), beta_nn_input (94*94=P*N), factor_nn_input (94*1=P*1), labels (94, = N,)
        return align_df.index, align_df.values[:, :-1].T, factor_nn_input.T.values , align_df.values[:, -1].T
    
    
    def dataloader(self, period): 
        mon_list = pd.read_pickle('data/mon_list.pkl')
        mon_list = mon_list.loc[(mon_list >= period[0]) & (mon_list <= period[1])]
        beta_nn_input_set = []
        factor_nn_input_set = []
        label_set = []
        for mon in mon_list:
            _, _beta_input, _factor_input, label =  self._get_item(mon)
            beta_nn_input_set.append(_beta_input)
            factor_nn_input_set.append(_factor_input)
            label_set.append(label)
            
        beta_nn_input_set = torch.tensor(beta_nn_input_set, dtype=torch.float32).to(self.device)
        factor_nn_input_set = torch.tensor(factor_nn_input_set, dtype=torch.float32).to(self.device)
        label_set = torch.tensor(label_set, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(beta_nn_input_set, factor_nn_input_set, label_set)   
        return DataLoader(dataset, batch_size=1, shuffle=True)


    def forward(self, char, pfret):
        processed_char = self.beta_nn(char)
        processed_pfret = self.factor_nn(pfret)
        return torch.sum(processed_char * processed_pfret, dim=1)

    
    # train_one_epoch
    def __train_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
            # labels reshape: (1, 94) -> (94, ) (1*N => N,)
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)
            output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, labels)
            
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                # print(f'Batches: {i}, loss: {loss.item()}')
                pass

        return epoch_loss / len(self.train_dataloader)


    def __valid_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.valid_dataloader):
            # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
            # labels reshape: (1, 94) -> (94, ) (1*N => N,)
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)

            output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, labels)
            epoch_loss += loss.item()

        return epoch_loss / len(self.valid_dataloader)
    
    
    def train_model(self):
        if 'saved_models' not in os.listdir('./'):
            os.mkdir('saved_models')
        
        self.train_dataloader = self.dataloader(self.train_period)
        self.valid_dataloader = self.dataloader(self.valid_period)
        self.test_dataloader = self.dataloader(self.test_period)
        
        min_error = np.Inf
        no_update_steps = 0
        valid_loss = []
        train_loss = []
        for i in range(MAX_EPOCH):
            # print(f'Epoch {i}')
            self.train()
            train_error = self.__train_one_epoch()
            train_loss.append(train_error)
            
            self.eval()
            # valid and early stop
            with torch.no_grad():
                valid_error = self.__valid_one_epoch()
                
            valid_loss.append(valid_error)
            if valid_error < min_error:
                min_error = valid_error
                no_update_steps = 0
                # save model
                torch.save(self.state_dict(), f'./saved_models/{self.name}.pt')
            else:
                no_update_steps += 1
            
            if no_update_steps > 2: # early stop, if consecutive 3 epoches no improvement on validation set
                print(f'Early stop at epoch {i}')
                break
            # load from (best) saved model
            self.load_state_dict(torch.load(f'./saved_models/{self.name}.pt'))
        return train_loss, valid_loss
    
    
    def test_model(self):
        # beta, factor, label = self.test_dataset
        # i = np.random.randint(len(beta))
        # beta_nn_input = beta[i]
        # factor_nn_input = factor[i]
        # labels = label[i]
        output = None
        label = None
        for i, beta_nn_input, factor_nn_input, labels in enumerate(self.test_dataloader):
            # convert to tensor
            # beta_nn_input = torch.tensor(beta_nn_input, dtype=torch.float32).T.to(self.device)
            # factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device)
            # labels = torch.tensor(labels, dtype=torch.float32).T.to(self.device)
            output = self.forward(beta_nn_input, factor_nn_input)
            break

        loss = self.criterion(output, labels)
        print(f'Test loss: {loss.item()}')
        print(f'Predicted: {output}')
        print(f'Ground truth: {labels}')
        return output, labels


    def calBeta(self, month, skip_char=[]):
        _, beta_nn_input, _, _ = self._get_item(month) # beta input: 94*94 = P*N
        
        # if some variables need be omitted
        if len(skip_char):
            beta_nn_input = pd.DataFrame(beta_nn_input.T, columns=CHARAS_LIST) # N*P
            beta_nn_input[skip_char] = beta_nn_input[skip_char] * 0.0
            beta_nn_input = beta_nn_input.values.T # P*N
        
        beta_nn_input = torch.tensor(beta_nn_input, dtype=torch.float32).T.to(self.device) # N*P
        return self.beta_nn(beta_nn_input) # N*K
    
    
    def calFactor(self, month, skip_char=[]):
        _, _, factor_nn_input, _ = self._get_item(month) # factor input: P*1
        
        # if some variables need be omitted
        if len(skip_char):
            factor_nn_input = pd.DataFrame(factor_nn_input.T, columns=CHARAS_LIST) # 1*P
            factor_nn_input[skip_char] = factor_nn_input[skip_char] * 0.0
            factor_nn_input = factor_nn_input.values.T # P*1

        factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device) # 1*P
        factor_pred = self.factor_nn(factor_nn_input).T # K*1
        
        self.factor_nn_pred.append(factor_pred)
        
        return factor_pred # K*1
    
    
    def inference(self, month):
        if len(self.omit_char) == 0:
            assert month >= self.test_period[0], f"Month error, {month} is not in test period {self.test_period}"
            
            mon_factor, mon_beta = self.calFactor(month), self.calBeta(month)
            
            assert mon_beta.shape[1] == mon_factor.shape[0], f"Dimension mismatch between mon_factor: {mon_factor.shape} and mon_beta: {mon_beta.shape}"
            
            # R_{N*1} = Beta_{N*K} @ F_{K*1}
            return mon_beta @ mon_factor
        else:
            ret_R = []
            for char in self.omit_char:
                mon_factor, mon_beta = self.calFactor(month, [char]), self.calBeta(month, [char])
                ret_R.append((mon_beta @ mon_factor).cpu().detach().numpy()) # N*1
                
            mon_factor, mon_beta = self.calFactor(month), self.calBeta(month)
            ret_R.append((mon_beta @ mon_factor).cpu().detach().numpy()) # also add complete result
            
            return np.array(ret_R).squeeze(2).T # N*m
    
    
    def cal_delayed_Factor(self, month):
        # calculate the last day of the previous month
        if self.refit_cnt == 0:
            avg_f_pred = self.factor_nn_pred[0] # input of the first predict take hat{f}_t
            # print(avg_f_pred.shape)
        else:
            avg_f_pred = torch.mean(torch.stack(self.factor_nn_pred[:self.refit_cnt]), dim=0)

        return avg_f_pred
    
    
    def reset_weight(self):
        for layer in self.beta_nn: # reset beta_nn parameters
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for layer in self.factor_nn: # reset factor_nn parameters
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                
        self.optimizer.state = collections.defaultdict(dict) # reset optimizer state


    def release_gpu(self):
        if self.train_dataloader is not None:
            del self.train_dataloader
        if self.valid_dataloader is not None:
            del self.valid_dataloader
        if self.test_dataloader is not None:
            del self.test_dataloader
        torch.cuda.empty_cache()



class CA0(CA_base):
    def __init__(self, hidden_size, lr=0.001, omit_char=[], device='cuda'):
        CA_base.__init__(self, name=f'CA0_{hidden_size}', omit_char=omit_char, device=device)
        # P -> K
        self.beta_nn = nn.Sequential(
            # output layer
            nn.Linear(94, hidden_size)
        )
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)
        


class CA1(CA_base):
    def __init__(self, hidden_size, dropout=0.5, lr=0.001, omit_char=[], device='cuda'):
        CA_base.__init__(self, name=f'CA1_{hidden_size}', omit_char=omit_char, device=device)
        self.dropout = dropout
        # P -> 32 -> K
        self.beta_nn = nn.Sequential(
            # hidden layer 1
            nn.Linear(94, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # output layer
            nn.Linear(32, hidden_size)
        )
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)
        
        
        
class CA2(CA_base):
    def __init__(self, hidden_size, dropout=0.5, lr=0.001, omit_char=[], device='cuda'):
        CA_base.__init__(self, name=f'CA2_{hidden_size}', omit_char=omit_char, device=device)
        self.dropout = dropout
        # P -> 32 -> 16 -> K
        self.beta_nn = nn.Sequential(
            # hidden layer 1
            nn.Linear(94, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # hidden layer 2
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # output layer
            nn.Linear(16, hidden_size)
        )
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)



class CA3(CA_base):
    def __init__(self, hidden_size, dropout=0.5, lr=0.001, omit_char=[], device='cuda'):
        CA_base.__init__(self, name=f'CA3_{hidden_size}', omit_char=omit_char, device=device)
        self.dropout = dropout
        # P -> 32 -> 16 -> 8 -> K
        self.beta_nn = nn.Sequential(
            # hidden layer 1
            nn.Linear(94, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # hidden layer 2
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # hidden layer 3
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # output layer
            nn.Linear(8, hidden_size)
        )
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = nn.MSELoss().to(device)