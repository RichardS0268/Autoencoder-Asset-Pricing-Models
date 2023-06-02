import torch
from torch import nn
import pandas as pd
import numpy as np
from .modelBase import modelBase
from utils import charas
import datetime
from dateutil.relativedelta import relativedelta

MAX_EPOCH = 20
LEARNING_RATE = 1e-3

class CA_base(nn.Module, modelBase):
    def __init__(self, name, device='cuda'):
        nn.Module.__init__(self)
        modelBase.__init__(self, name)
        self.beta_nn = None
        self.factor_nn = None
        self.optimizer = None
        self.criterion = None

        self.device = device

        self.datashare_chara = pd.read_pickle('./data/datashare_re.pkl').astype(np.float64)
        self.portfolio_ret=  pd.read_pickle('./data/portfolio_ret.pkl').astype(np.float64)
        self.mon_ret = pd.read_pickle('./data/month_ret.pkl').astype(np.float64)

    def _get_item(self, month):
        beta_nn_input = self.datashare_chara.loc[self.datashare_chara['DATE'] == month].set_index('permno')[charas]
        labels = self.mon_ret.loc[self.mon_ret['date'] == month].drop_duplicates('permno').set_index('permno')['ret-rf']
        align_df = pd.concat([beta_nn_input, labels], axis=1).dropna()
        factor_nn_input = self.portfolio_ret.loc[self.portfolio_ret['DATE'] == month][charas].T.values     
        # exit(0) if there is any nan in align_df
        if align_df.isnull().values.any():
            assert False, f'There is nan in align_df of : {month}'
        # return stock index (N), beta_nn_input (94*N), factor_nn_input (94*1), labels (N, )
        return align_df.index, align_df.values[:, :-1].T, factor_nn_input, align_df.values[:, -1].T
    
    
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
            
        ##TODO: tensorize, shuffle (synchronously)
        ## shuffle will be implemented inside of train_one_epoch
        return beta_nn_input_set, factor_nn_input_set, label_set

    def forward(self, char, pfret):
        processed_char = self.beta_nn(char)
        processed_pfret = self.factor_nn(pfret)
        # dot product of two processed tensors
        return torch.sum(processed_char * processed_pfret, dim=1)

    
    ##TODO: train_one_epoch
    def __train_one_epoch(self):
        beta_nn_input_set, factor_nn_input_set, label_set = self.train_dataset
        shuffled_ind = np.random.permutation(len(beta_nn_input_set))
        epoch_loss = 0.0
        for i, ind in enumerate(shuffled_ind):
            beta_nn_input = beta_nn_input_set[ind]
            factor_nn_input = factor_nn_input_set[ind]
            labels = label_set[ind]
            # convert to tensor
            beta_nn_input = torch.tensor(beta_nn_input, dtype=torch.float32).T.to(self.device)
            factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device)
            labels = torch.tensor(labels, dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()
            output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, labels)
            
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                # print(f'Batches: {i}, loss: {loss.item()}')
                pass
        return epoch_loss / len(beta_nn_input_set)

    def __valid_one_epoch(self):
        # run validation, no gradient calculation
        beta_nn_input_set, factor_nn_input_set, label_set = self.valid_dataset
        epoch_loss = 0.0
        for i in range(len(beta_nn_input_set)):
            beta_nn_input = beta_nn_input_set[i]
            factor_nn_input = factor_nn_input_set[i]
            labels = label_set[i]

            # convert to tensor
            beta_nn_input = torch.tensor(beta_nn_input, dtype=torch.float32).T.to(self.device)
            factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device)
            labels = torch.tensor(labels, dtype=torch.float32).T.to(self.device)

            output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, labels)
            epoch_loss += loss.item()
        return epoch_loss / len(beta_nn_input_set)
    
    def train_model(self):
        self.train_dataset = self.dataloader(self.train_period)
        self.valid_dataset = self.dataloader(self.valid_period)
        self.test_dataset = self.dataloader(self.test_period)
        
        min_error = np.Inf
        no_update_steps = 0
        valid_loss = []
        for i in range(MAX_EPOCH):
            print(f'Epoch {i}')
            self.train()
            self.__train_one_epoch()
            
            self.eval()
            ##TODO, valid and early stop
            valid_error = self.__valid_one_epoch()
            valid_loss.append(valid_error)
            if valid_error < min_error:
                min_error = valid_error
                no_update_steps = 0
            else:
                no_update_steps += 1
            
            if no_update_steps > 15: # early stop
                print(f'Early stop at epoch {i}')
                break
        return valid_loss
    
    def test_model(self):
        beta, factor, label = self.test_dataset
        i = np.random.randint(len(beta))
        beta_nn_input = beta[i]
        factor_nn_input = factor[i]
        labels = label[i]

        # convert to tensor
        beta_nn_input = torch.tensor(beta_nn_input, dtype=torch.float32).T.to(self.device)
        factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device)
        labels = torch.tensor(labels, dtype=torch.float32).T.to(self.device)

        output = self.forward(beta_nn_input, factor_nn_input)
        loss = self.criterion(output, labels)
        print(f'Test loss: {loss.item()}')
        print(f'Predicted: {output}')
        print(f'Ground truth: {labels}')
        return output, labels


    def calBeta(self, month):
        _, beta_nn_input, _, _ = self._get_item(month)
        
        beta_nn_input = torch.tensor(beta_nn_input, dtype=torch.float32).T.to(self.device)
        return self.beta_nn(beta_nn_input)
    
    
    def calFactor(self, month):
        _, _, factor_nn_input, _ = self._get_item(month)

        factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device)
        return self.factor_nn(factor_nn_input)
    
    def cal_delayed_Factor(self, month):
        # calculate the last day of previous month
        prev_month = datetime.datetime.strptime(str(month), '%Y%m%d') - relativedelta(months=1)
        prev_month = int(prev_month.strftime('%Y%m%d'))
        _, _, factor_nn_input, _ = self._get_item(prev_month)

        factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device)
        return self.factor_nn(factor_nn_input)
    
class CA0(CA_base):
    def __init__(self, hidden_size, lr=0.001, device='cuda'):
        CA_base.__init__(self, f'CA0_{hidden_size}', device=device)
        # P -> hidden_size
        self.beta_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)
        


class CA1(CA_base):
    def __init__(self, hidden_size, dropout, lr, device='cuda'):
        CA_base.__init__(self, f'CA1_{hidden_size}', device=device)
        self.dropout = dropout
        # P -> hidden_size
        self.beta_nn = nn.Sequential(
            nn.Linear(94, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(32, hidden_size)
        )
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)
        
        
        
class CA2(CA_base):
    def __init__(self, hidden_size, dropout=0.5, lr=0.001, device='cuda'):
        CA_base.__init__(self, f'CA2_{hidden_size}', device=device)
        self.dropout = dropout
        # P -> 32 -> hidden_size
        self.beta_nn = nn.Sequential(
            nn.Linear(94, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(16, hidden_size)
        )
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)



class CA3(CA_base):
    def __init__(self, hidden_size, dropout=0.5, lr=0.001, device='cuda'):
        CA_base.__init__(self, f'CA3_{hidden_size}', device=device)
        # P -> 32 -> 16 -> 8
        self.dropout = dropout

        self.beta_nn = nn.Sequential(
            nn.Linear(94, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(8, hidden_size)
        )
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)
