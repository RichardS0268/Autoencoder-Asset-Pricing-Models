import torch
from torch import nn
import pandas as pd
import numpy as np
from .modelBase import modelBase
from utils import charas

MAX_EPOCH = 1
LEARNING_RATE = 1e-3

class CA_base(modelBase):
    def __init__(self, name):
        super().__init__(name)
        self.beta_nn = None
        self.factor_nn = None
        self.optimizer = None
        self.criterion = None

        self.datashare_chara = pd.read_pickle('data/datashare_re.pkl')
        self.portfolio_ret=  pd.read_pickle('data/portfolio_ret.pkl')
        self.mon_ret = pd.read_pickle('data/month_ret.pkl')

    def __get_item(self, month):
        beta_nn_input = self.datashare_chara.loc[self.datashare_chara['DATE'] == month].set_index('permno')[charas]
        labels = self.mon_ret.loc[self.mon_ret['date'] == month].set_index('permno')['ret-rf']
        align_df = pd.concat([beta_nn_input, labels], axis=1)
        factor_nn_input = self.portfolio_ret.loc[self.portfolio_ret['DATE'] == month][charas].T.values        
        
        # return stock index (N), beta_nn_input (94*N), factor_nn_input (94*1), labels (N, )
        return align_df.index, align_df.values[:, :-1].T, factor_nn_input, align_df.values[:, -1].T
    
    
    def dataloader(self, period): 
        mon_list = pd.read_pickle('data/mon_list.pkl')
        mon_list = mon_list.loc[(mon_list >= period[0]) & (mon_list <= period[1])]
        beta_nn_input_set = []
        factor_nn_input_set = []
        label_set = []
        for mon in mon_list:
            _, _beta_input, _factor_input, label =  self.__get_item(mon)
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

            self.optimizer.zero_grad()
            output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f'Batches: {i}, loss: {loss.item()}')
        return epoch_loss

    def __valid_one_epoch(self):
        # run validation, no gradient calculation
        beta_nn_input_set, factor_nn_input_set, label_set = self.valid_dataset
        epoch_loss = 0.0
        for i in range(len(beta_nn_input_set)):
            beta_nn_input = beta_nn_input_set[i]
            factor_nn_input = factor_nn_input_set[i]
            labels = label_set[i]

            output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, labels)
            epoch_loss += loss.item()
        return epoch_loss
    
    def train_model(self):
        self.train_dataset = self.dataloader(self.train_period)
        self.valid_dataset = self.dataloader(self.valid_period)
        
        min_error = np.Inf
        no_update_steps = 0
        valid_loss = []
        for i in range(MAX_EPOCH):
            self.__train_one_epoch()
            
            ##TODO, valid and early stop
            valid_error = self.__valid_one_epoch()
            valid_loss.append(valid_error)
            if valid_error < min_error:
                min_error = valid_error
                no_update_steps = 0
            else:
                no_update_steps += 1
            
            if no_update_steps > 2: # early stop
                break
        
        return valid_loss
    
    def calBeta(self, month):
        _, beta_nn_input, _, _ = self.__get_item(month)
        return self.beta_nn(beta_nn_input)
    
    
    def calFactor(self, month):
        _, _, factor_nn_input, _ = self.__get_item(month)
        return self.factor_nn(factor_nn_input)
    
    
    
    
class CA0(CA_base, nn.Module):
    def __init__(self, hidden_size):
        super(CA0, self).__init__('CA0')
        # P -> hidden_size
        self.beta_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )
        self.factor_nn = nn.Sequential(
            nn.Linear(94, hidden_size)
        )
        


class CA1(CA_base, nn.Module):
    def __init__(self, hidden_size, dropout):
        super(CA1, self).__init__('CA1')
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
        
        
        
class CA2(CA_base, nn.Module):
    def __init__(self, hidden_size, dropout):
        super(CA2, self).__init__('CA2')
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



class CA3(nn.Module, CA_base):
    def __init__(self, hidden_size, dropout):
        # init nn.Module 
        nn.Module.__init__(self)
        CA_base.__init__(self, 'CA3')
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
        

# def epoch_train(model, train_loader, optimizer, criterion, epoch):
#     # train model for one epoch
#     model.train()
#     train_loss = 0
#     for batch_idx, (exposure, factor, label) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(exposure, factor)
#         loss = criterion(output, label)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
#                 batch_idx * len(exposure),
#                 len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.item()))
#     print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


# def epoch_val(model, val_loader, criterion):
#     # validate model for one epoch
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for exposure, factor, label in val_loader:
#             output = model(exposure, factor)
#             val_loss += criterion(output, label).item()
#     val_loss /= len(val_loader.dataset)
#     print('====> Validation set loss: {:.4f}'.format(val_loss))
#     return val_loss


# def test(model, test_loader):
#     # test model
#     model.eval()
#     with torch.no_grad():
#         for exposure, factor, label in test_loader:
#             output = model(exposure, factor)
#             print(output)
#             print(label)
#             break


# def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
#     # train model for num_epochs
#     train_loss_list = []
#     val_loss_list = []
#     for epoch in range(1, num_epochs + 1):
#         epoch_train(model, train_loader, optimizer, criterion, epoch)
#         val_loss = epoch_val(model, val_loader, criterion)
#         train_loss_list.append(epoch)
#         val_loss_list.append(val_loss)
#     return train_loss_list, val_loss_list

def main():
    # generate random data to train and test model
    # batch size is variable N
    # exposure is a tensor of shape (N, 94)
    # factor is a tensor of shape (N, 94)
    N = 100
    character = torch.randn(N, 94)
    pfret = torch.randn(N, 94)
    label = torch.randn(N)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(character, pfret, label), batch_size=N, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(character, pfret, label), batch_size=N, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(character, pfret, label), batch_size=N, shuffle=True)

    # initialize model
    model = CA3(hidden_size=8, dropout=0.5)
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # initialize loss function
    criterion = nn.MSELoss()

    # train model
    train_loss_list, val_loss_list = train(model, train_loader, val_loader, optimizer, criterion, 10)
    # test model
    test(model, test_loader)

if __name__ == '__main__':
    main()
