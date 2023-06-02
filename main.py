from models.CA import CA3
import torch
from torch import nn
from matplotlib import pyplot as plt

model = CA3(6, 0.5)
model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.criterion = nn.MSELoss()

valid_loss = model.train_model()
print(valid_loss)