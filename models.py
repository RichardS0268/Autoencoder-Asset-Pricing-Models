import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # P -> 32 -> 16 -> 8
        self.exposure_nn = nn.Sequential(
            nn.Linear(94, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.factor_nn = nn.Sequential(
            nn.Linear(94, 8),
        )

    def forward(self, exposure, factor):
        processed_exposure = self.exposure_nn(exposure)
        processed_factor = self.factor_nn(factor)
        # dot product of two processed tensors
        return torch.sum(processed_exposure * processed_factor, dim=1)

def epoch_train(model, train_loader, optimizer, criterion, epoch):
    # train model for one epoch
    model.train()
    train_loss = 0
    for batch_idx, (exposure, factor, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(exposure, factor)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                batch_idx * len(exposure),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def epoch_val(model, val_loader, criterion):
    # validate model for one epoch
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for exposure, factor, label in val_loader:
            output = model(exposure, factor)
            val_loss += criterion(output, label).item()
    val_loss /= len(val_loader.dataset)
    print('====> Validation set loss: {:.4f}'.format(val_loss))
    return val_loss

def test(model, test_loader):
    # test model
    model.eval()
    with torch.no_grad():
        for exposure, factor, label in test_loader:
            output = model(exposure, factor)
            print(output)
            print(label)
            break

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    # train model for num_epochs
    train_loss_list = []
    val_loss_list = []
    for epoch in range(1, num_epochs + 1):
        epoch_train(model, train_loader, optimizer, criterion, epoch)
        val_loss = epoch_val(model, val_loader, criterion)
        train_loss_list.append(epoch)
        val_loss_list.append(val_loss)
    return train_loss_list, val_loss_list

def main():
    # generate random data to train and test model
    # batch size is variable N
    # exposure is a tensor of shape (N, 94)
    # factor is a tensor of shape (N, 94)
    N = 100
    exposure = torch.randn(N, 94)
    factor = torch.randn(N, 94)
    label = torch.randn(N)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(exposure, factor, label), batch_size=N, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(exposure, factor, label), batch_size=N, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(exposure, factor, label), batch_size=N, shuffle=True)

    # initialize model
    model = Model()
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
