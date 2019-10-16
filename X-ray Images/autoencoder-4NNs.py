import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import os
from torchvision import transforms
from torch.nn import DataParallel
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
ids = [0, 1]
torch.cuda.empty_cache()
# initial values
EPOCH = 50
BATCH_SIZE = 64
LR = 0.0005

# load images data
train_data = torchvision.datasets.ImageFolder('./Training_set', transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]))
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# NN network
class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1, self).__init__()
        self.encoder = nn.Sequential(  # two layers encoder
            nn.Linear(74*74, 8000),
            nn.ReLU(True),  # ReLU, Tanh, etc.
            nn.Linear(8000, 1000),
            nn.ReLU(True),  # input is in (0,1), so select this one
        )
        self.decoder = nn.Sequential(  # two layers decoder
            nn.Linear(1000, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 74*74),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder2, self).__init__()
        self.encoder = nn.Sequential(  # two layers encoder
            nn.Linear(74*74, 8000),
            nn.ReLU(True),  # ReLU, Tanh, etc.
            nn.Linear(8000, 1000),
            nn.ReLU(True),  # input is in (0,1), so select this one
        )
        self.decoder = nn.Sequential(  # two layers decoder
            nn.Linear(1000, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 74*74),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder3(nn.Module):
    def __init__(self):
        super(Autoencoder3, self).__init__()
        self.encoder = nn.Sequential(  # two layers encoder
            nn.Linear(74*74, 8000),
            nn.ReLU(True),  # ReLU, Tanh, etc.
            nn.Linear(8000, 1000),
            nn.ReLU(True),  # input is in (0,1), so select this one
        )
        self.decoder = nn.Sequential(  # two layers decoder
            nn.Linear(1000, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 74*74),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder4(nn.Module):
    def __init__(self):
        super(Autoencoder4, self).__init__()
        self.encoder = nn.Sequential(  # two layers encoder
            nn.Linear(74*74, 8000),
            nn.ReLU(True),  # ReLU, Tanh, etc.
            nn.Linear(8000, 1000),
            nn.ReLU(True),  # input is in (0,1), so select this one
        )
        self.decoder = nn.Sequential(  # two layers decoder
            nn.Linear(1000, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 74*74),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder1 = DataParallel(Autoencoder1()).cuda()
autoencoder2 = DataParallel(Autoencoder2()).cuda()
autoencoder3 = DataParallel(Autoencoder3()).cuda()
autoencoder4 = DataParallel(Autoencoder4()).cuda()
optimizer1 = torch.optim.Adam(autoencoder1.parameters(),
                              lr=LR)  # optimizer: SGD, Momentum, Adagrad, etc. This one is better.
optimizer2 = torch.optim.Adam(autoencoder2.parameters(), lr=LR)
optimizer3 = torch.optim.Adam(autoencoder3.parameters(), lr=LR)
optimizer4 = torch.optim.Adam(autoencoder4.parameters(), lr=LR)
loss_func = nn.MSELoss()  # loss function: MSE

for epoch in range(EPOCH):
    for step, (x, x_label) in enumerate(train_loader):  # train_loader has the number of batches, data, and label
        b_x = x.view(-1, 128 * 128)  # input data
        b_y = x.view(-1, 128 * 128).cuda()  # comparing data
        b_xx = b_x.view(128,128)
        b_x1 = torch.narrow(torch.narrow(b_xx, 1, 0, 74),0,0,74).view(-1,74*74).cuda()
        b_x2 = torch.narrow(torch.narrow(b_xx, 1, 54, 128),0,0,74).view(-1,74*74).cuda()
        b_x3 = torch.narrow(torch.narrow(b_xx, 1, 0, 74),0,54,128).view(-1,74*74).cuda()
        b_x4 = torch.narrow(torch.narrow(b_xx, 1, 54, 128),0,54,128).view(-1,74*74).cuda()
        # running in the neural network
        decoded1 = autoencoder1(b_x1)
        decoded2 = autoencoder2(b_x2)
        decoded3 = autoencoder3(b_x3)
        decoded4 = autoencoder4(b_x4)
        # concatenate 4 parts
        decoded1_se = torch.narrow(decoded1, 1, 0, 64)
        decoded1_co = torch.narrow(decoded1, 1, 64,74)
        decoded2_se = torch.narrow(decoded2, 1, 10, 74)
        decoded2_co = torch.narrow(decoded2, 1, 0, 10)
        decoded3_se = torch.narrow(decoded3, 1, 0, 64)
        decoded3_co = torch.narrow(decoded3, 1, 64, 74)
        decoded4_se = torch.narrow(decoded4, 1, 10, 74)
        decoded4_co = torch.narrow(decoded4, 1, 0, 10)
        decoded12_ave = (decoded1_co+decoded2_co)/2
        decoded34_ave = (decoded3_co+decoded4_co)/2
        up_part = torch.cat([decoded1_se,decoded12_ave,decoded2_se],dim=1)
        down_part = torch.cat([decoded3_se, decoded34_ave, decoded4_se], dim=1)
        up_part_se = torch.narrow(up_part, 0, 0, 64)
        up_part_co = torch.narrow(up_part, 0, 64, 74)
        down_part_se = torch.narrow(down_part, 0, 10, 74)
        down_part_co = torch.narrow(down_part, 0, 0, 10)
        updown_ave = (up_part_co+down_part_co)/2
        decoded = torch.cat([up_part_se,updown_ave,down_part_se], dim=0)

        loss = loss_func(decoded, b_y)
        optimizer1.zero_grad()  # initialize the optimizer
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        if step % 50 == 0:
            print('Epoch:', epoch, '| tran loss : %.4f' % loss.data.cpu().numpy(), 'count:', len(b_x[1, :]))

torch.save(autoencoder1.module.state_dict(), '4NNs-part1.pkl')  # save the parameter values of neural network
torch.save(autoencoder2.module.state_dict(), '4NNs-part2.pkl')
torch.save(autoencoder3.module.state_dict(), '4NNs-part3.pkl')
torch.save(autoencoder4.module.state_dict(), '4NNs-part4.pkl')
