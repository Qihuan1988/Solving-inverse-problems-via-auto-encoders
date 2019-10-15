import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms
import scipy.io as sio

test_data = torchvision.datasets.ImageFolder('./Testing_set', transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))

test_loader=Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

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

#Load 4 NNs
autoencoder1=Autoencoder1()
autoencoder1.load_state_dict(torch.load('4NNs-part1.pkl', map_location=lambda storage, loc:storage)) #load the parameter values of NN
autoencoder2=Autoencoder2()
autoencoder2.load_state_dict(torch.load('4NNs-part2.pkl', map_location=lambda storage, loc:storage))
autoencoder3=Autoencoder3()
autoencoder3.load_state_dict(torch.load('4NNs-part3.pkl', map_location=lambda storage, loc:storage))
autoencoder4=Autoencoder4()
autoencoder4.load_state_dict(torch.load('4NNs-part4.pkl', map_location=lambda storage, loc:storage))

n=128*128 #picture size
m=np.array([np.int(n*0.005),np.int(n*0.01),np.int(n*0.03), np.int(n*0.05),np.int(n*0.1),np.int(n*0.2)]) #compressed picture size
test_images=100
sumPSNR=np.zeros(len(m))
for cc in range(0,len(m)):
    #get the matrix A
    M_no=cc+1
    data1 = sio.loadmat('./MatrixA/A%s'%M_no)
    M = data1['M']
    A1=M
    A=torch.from_numpy(A1).float()
    At=torch.from_numpy(np.transpose(A1))
    loss_func=nn.MSELoss()
    ss=0
    #gradient descent
    for step, (x,b_label) in enumerate(test_loader):
        b_x = x.view(-1, 128 * 128)
        y = torch.mm(A,torch.t(b_x)).detach()
        test_y = y
        s_hat=torch.mm(torch.t(A),test_y)
        s_hat=torch.t(s_hat).view(128,128) #stand for s in the paper
        mse_err_p=1e6
        mse_step=0
        umin=0.7
        ss = ss + 1
        vv = 1
        while (True):
            s_hat1 = torch.narrow(torch.narrow(s_hat, 1, 0, 74), 0, 0, 74).contiguous().view(-1, 74 * 74)
            s_hat2 = torch.narrow(torch.narrow(s_hat, 1, 54, 74), 0, 0, 74).contiguous().view(-1, 74 * 74)
            s_hat3 = torch.narrow(torch.narrow(s_hat, 1, 0, 74), 0, 54, 74).contiguous().view(-1, 74 * 74)
            s_hat4 = torch.narrow(torch.narrow(s_hat, 1, 54, 74), 0, 54, 74).contiguous().view(-1, 74 * 74)
            decoded1 = autoencoder1(s_hat1).view(74,74)
            decoded2 = autoencoder2(s_hat2).view(74,74)
            decoded3 = autoencoder3(s_hat3).view(74,74)
            decoded4 = autoencoder4(s_hat4).view(74,74)
            decoded1_se = torch.narrow(decoded1, 1, 0, 54)
            decoded1_co = torch.narrow(decoded1, 1, 54, 20)
            decoded2_se = torch.narrow(decoded2, 1, 20, 54)
            decoded2_co = torch.narrow(decoded2, 1, 0, 20)
            decoded3_se = torch.narrow(decoded3, 1, 0, 54)
            decoded3_co = torch.narrow(decoded3, 1, 54, 20)
            decoded4_se = torch.narrow(decoded4, 1, 20, 54)
            decoded4_co = torch.narrow(decoded4, 1, 0, 20)
            decoded12_ave = (decoded1_co + decoded2_co) / 2
            decoded34_ave = (decoded3_co + decoded4_co) / 2
            up_part = torch.cat([decoded1_se, decoded12_ave, decoded2_se], dim=1)
            down_part = torch.cat([decoded3_se, decoded34_ave, decoded4_se], dim=1)
            up_part_se = torch.narrow(up_part, 0, 0, 54)
            up_part_co = torch.narrow(up_part, 0, 54, 20)
            down_part_se = torch.narrow(down_part, 0, 20, 54)
            down_part_co = torch.narrow(down_part, 0, 0, 20)
            updown_ave = (up_part_co + down_part_co) / 2
            decoded = torch.cat([up_part_se, updown_ave, down_part_se], dim=0)
            x_hat = decoded.view(128 * 128,1
            tem = test_y - torch.mm(A, x_hat)
            gradient=torch.mm(torch.t(A),tem)
            compare=x_hat+umin*gradient
            compare=torch.t(compare)
            mse_err=loss_func(test_y,torch.mm(A,torch.t(compare)).detach())
            if mse_err>mse_err_p:
                mse_step=mse_step+1
                if mse_step==1:
                    output=x_hat.view(-1,128*128)
                if mse_step>3:
                    break
            else:
                mse_step=0
                vv = vv + 1
                mse_err_p=mse_err
            s_hat=compare.view(128,128)
        #calculate PSNR
        iterations[cc,step]=vv
        MSE=loss_func(output,b_x)
        xxx=torch.max(b_x)**2/MSE
        PSNR=10*torch.log10(xxx)
        sumPSNR[cc]=PSNR+sumPSNR[cc]
        #np.savetxt('./Reconstructed Images/4NNs-PGD/%s/imageoutput%s.txt'%(cc,step),output.detach().numpy())
    print('PSNR:',sumPSNR[cc]/test_images)

