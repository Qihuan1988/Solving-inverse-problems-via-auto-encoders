import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import os
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
ids = [0]
torch.cuda.empty_cache()
# initial values
EPOCH = 200 #This can give a good result
BATCH_SIZE = 128
LR = 0.0001

# load images data
train_data = torchvision.datasets.ImageFolder('./Training_set', transform=transforms.Compose([
    transforms.ToTensor()]))
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# NN network
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # two layers encoder
            nn.Linear(64 * 64*3, 12000),
            nn.Sigmoid(),  # ReLU, Tanh, etc.
             # input is in (0,1), so select this one
            nn.Linear(12000, 3000),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(  # two layers decoder
            nn.Linear(3000, 12000),
            nn.Sigmoid(),
            nn.Linear(12000, 64 * 64*3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = Autoencoder().cuda()
optimizer = torch.optim.Adam(autoencoder.parameters(),
                              lr=LR)  # optimizer: SGD, Momentum, Adagrad, etc. This one is better.
loss_func = nn.MSELoss()  # loss function: MSE

for epoch in range(EPOCH):
    for step, (x, x_label) in enumerate(train_loader):  # train_loader has the number of batches, data, and label
        # print(x.size())
        b_x = x.view(-1, 64 * 64*3).cuda()  # input data
        b_y = x.view(-1, 64 * 64*3).cuda()  # comparing data
        # running in the neural network
        encoded, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()  # initialize the optimizer
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print('Epoch:', epoch, '| tran loss : %.4f' % loss.data.cpu().numpy())
        # We don't provide the average training error and validation error
torch.save(autoencoder.state_dict(), 'AE-trained.pkl')  # save the parameter values of neural network
