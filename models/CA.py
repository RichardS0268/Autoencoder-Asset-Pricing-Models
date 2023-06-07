import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from .modelBase import modelBase
from utils import charas
import datetime
import calendar
from dateutil.relativedelta import relativedelta

from models.Attentions import *

MAX_EPOCH = 200

class CA_base(nn.Module, modelBase):
    def __init__(self, name, device='cuda', portfolio=True):
        nn.Module.__init__(self)
        modelBase.__init__(self, name)
        self.beta_nn = None
        self.factor_nn = None
        self.optimizer = None
        self.criterion = None
        self.portfolio = portfolio

        self.device = device

        self.datashare_chara = pd.read_pickle('./data/datashare_re.pkl').astype(np.float64)
        self.p_charas = pd.read_pickle('./data/p_charas.pkl').astype(np.float64).reset_index()
        self.portfolio_ret=  pd.read_pickle('./data/portfolio_ret.pkl').astype(np.float64)
        self.mon_ret = pd.read_pickle('./data/month_ret.pkl').astype(np.float64)

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
    
    def debug(self, month):
        beta_nn_input = self.p_charas.loc[self.p_charas['DATE'] == month][charas]
        # beta_nn_input = self.datashare_chara.loc[self.datashare_chara['DATE'] == month].set_index('permno')[charas]
        print(beta_nn_input)

    def _get_item(self, month):
        if not self.portfolio:
            if month not in self.datashare_chara['DATE'].values:
                # find the closest month in datashare_chara to month
                month = self.datashare_chara['DATE'].values[np.argmin(np.abs(self.datashare_chara['DATE'].values - month))]
            beta_nn_input = self.datashare_chara.loc[self.datashare_chara['DATE'] == month].set_index('permno')[charas]
            labels = self.mon_ret.loc[self.mon_ret['date'] == month].drop_duplicates('permno').set_index('permno')['ret-rf']
            align_df = pd.concat([beta_nn_input, labels], axis=1).dropna()
        else: 
            if month not in self.p_charas['DATE'].values:
                # find the closest month in p_charas to month
                month = self.p_charas['DATE'].values[np.argmin(np.abs(self.p_charas['DATE'].values - month))]
            beta_nn_input = self.p_charas.loc[self.p_charas['DATE'] == month][charas] # (94, 94)
            labels = self.portfolio_ret.loc[self.portfolio_ret['DATE'] == month][charas].T.values # (94, 1)
            beta_nn_input['ret-rf'] = labels
            align_df = beta_nn_input.copy(deep=False).dropna()
            
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
            
        beta_nn_input_set = torch.tensor(beta_nn_input_set, dtype=torch.float32).to(self.device)
        factor_nn_input_set = torch.tensor(factor_nn_input_set, dtype=torch.float32).to(self.device)
        label_set = torch.tensor(label_set, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(beta_nn_input_set, factor_nn_input_set, label_set)   
        return DataLoader(dataset, batch_size=1, shuffle=True)




    def forward(self, char, pfret):
        processed_char = self.beta_nn(char)
        processed_pfret = self.factor_nn(pfret)
        return torch.sum(processed_char * processed_pfret, dim=1)

    
    ##TODO: train_one_epoch
    def __train_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels)  in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            # beta_nn_input reshape: (1, 94, N) -> (N, 94)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94)
            # labels reshape: (1, N) -> (N, )
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
            # beta_nn_input reshape: (1, 94, N) -> (N, 94)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94)
            # labels reshape: (1, N) -> (N, )
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)

            output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, labels)
            epoch_loss += loss.item()

        return epoch_loss / len(self.valid_dataloader)
    
    def train_model(self):
        self.train_dataloader = self.dataloader(self.train_period)
        self.valid_dataloader = self.dataloader(self.valid_period)
        self.test_dataloader = self.dataloader(self.test_period)
        
        min_error = np.Inf
        no_update_steps = 0
        valid_loss = []
        train_loss = []
        for i in range(MAX_EPOCH):
            print(f'Epoch {i}')
            self.train()
            train_error = self.__train_one_epoch()
            train_loss.append(train_error)
            
            self.eval()
            ##TODO, valid and early stop
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
            
            if no_update_steps > 20: # early stop
                print(f'Early stop at epoch {i}')
                break
            # load from saved model
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


    def calBeta(self, month):
        _, beta_nn_input, _, _ = self._get_item(month)
        
        beta_nn_input = torch.tensor(beta_nn_input, dtype=torch.float32).T.to(self.device)
        return self.beta_nn(beta_nn_input)
    
    
    def calFactor(self, month):
        _, _, factor_nn_input, _ = self._get_item(month)

        factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device)
        return self.factor_nn(factor_nn_input).T
    
    def cal_delayed_Factor(self, month):
        # calculate the last day of the previous month
        prev_month = datetime.datetime.strptime(str(month), '%Y%m%d') - relativedelta(months=1)
        prev_month = datetime.datetime(prev_month.year, prev_month.month, calendar.monthrange(prev_month.year, prev_month.month)[1]).strftime('%Y%m%d')
        prev_month = int(prev_month)
        
        _, _, factor_nn_input, _ = self._get_item(prev_month)

        factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device)
        return self.factor_nn(factor_nn_input).T
    
    def reset_weight(self):
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_weight)

    def release_gpu(self):
        if self.train_dataloader is not None:
            del self.train_dataloader
        if self.valid_dataloader is not None:
            del self.valid_dataloader
        if self.test_dataloader is not None:
            del self.test_dataloader
        torch.cuda.empty_cache()

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
    def __init__(self, hidden_size, dropout=0.5, lr=0.001, device='cuda'):
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

class CAA2(CA_base):
    def __init__(self, hidden_size, dropout=0.5, lr=0.001, device="cuda"):
        CA_base.__init__(self, f'CAA2_{hidden_size}', device=device)
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
        self.beta_attention = CrossAttention(hidden_size, hidden_size, self.dropout)
        self.factor_attention = CrossAttention(hidden_size, hidden_size, self.dropout)
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(device)
    
    def forward(self, char, pfret, query):
        beta = self.beta_nn(char) # (N, hidden_size)
        factor = self.factor_nn(pfret) # (1, hidden_size)
        # get dot product 
        # query = torch.matmul(beta, factor.T) # (N, 1)
        # get attention
        # print(beta.shape, factor.shape, query.shape)
        # by now beta is (N, hidden_size), factor is (1, hidden_size), query is (N, 1)
        # convert beta to (N, hidden_size, 1), factor to (1, hidden_size, 1)
        query = query.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        factor = factor.unsqueeze(-1)
        beta_a = self.beta_attention(query, beta) # (N, hidden_size)
        factor_a = self.factor_attention(query, factor) # (1, hidden_size)

        # print(beta_a.shape, factor_a.shape)

        return torch.sum(beta_a * factor_a, dim=1)  # (N, 1) -> (N, )

    def __train_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels)  in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            # beta_nn_input reshape: (1, 94, N) -> (N, 94)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94)
            # labels reshape: (1, N) -> (N, )
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)
            output = self.forward(beta_nn_input, factor_nn_input, labels.unsqueeze(-1))
            # print(output.shape, labels.shape)
            loss = self.criterion(output, labels)
            
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                # print(f'Batches: {i}, loss: {loss.item()}')
                pass

        return epoch_loss / len(self.train_dataloader)

    def train_model(self):
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
            ##TODO, valid and early stop
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
            
            if no_update_steps > 20: # early stop
                print(f'Early stop at epoch {i}')
                break
            # load from saved model
            self.load_state_dict(torch.load(f'./saved_models/{self.name}.pt'))
        return train_loss, valid_loss
    
    def __valid_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.valid_dataloader):
            # beta_nn_input reshape: (1, 94, N) -> (N, 94)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94)
            # labels reshape: (1, N) -> (N, )
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)

            output = self.forward(beta_nn_input, factor_nn_input, labels.unsqueeze(-1))
            loss = self.criterion(output, labels)
            epoch_loss += loss.item()

        return epoch_loss / len(self.valid_dataloader)