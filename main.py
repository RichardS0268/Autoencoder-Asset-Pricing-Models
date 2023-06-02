from models.CA import CA3
import torch
from torch import nn
from matplotlib import pyplot as plt

model = CA3(6, 0.5).to('cuda')
# model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# model.criterion = nn.MSELoss().to('cuda')

train_loss, valid_loss = model.train_model()
plt.plot(train_loss, label='train') 
plt.plot(valid_loss, label='valid')
plt.legend()
plt.show()

