import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import scipy.io as sio

#Load date from test data document
test_data = torchvision.datasets.ImageFolder('./Testing_set', transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
test_loader=Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

#to load the trained neural network (same structures as the trained NN)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder=nn.Sequential( #two layers encoder
            nn.Linear(128*128,5000),
            nn.ReLU(True), #ReLU, Tanh, etc.
            nn.Linear(5000,2000),
            nn.ReLU(True) #input is in (0,1), so select this one
        )
        self.decoder=nn.Sequential( #two layers decoder
            nn.Linear(2000, 5000),
            nn.ReLU(True),
            nn.Linear(5000, 128*128),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded

#Load the trained NN parameters to the new NN
autoencoder=Autoencoder()
autoencoder.load_state_dict(torch.load('NN1-Xray.pkl', map_location=lambda storage, loc:storage)) #load the parameter values of NN

n=128*128 #picture size
m=np.array([np.int(n*0.005),np.int(n*0.01),np.int(n*0.03), np.int(n*0.05),np.int(n*0.1),np.int(n*0.2)]) #compressed picture size
test_images=100
sumPSNR=np.zeros(len(m)) #total PSNR
Noise_exist=False
if Noise_exist==True:
    start_num=3
else:
    start_num=0
for cc in range(start_num,len(m)): #for different m values
    #get the matrix A
    M_no=cc+1
    data1 = sio.loadmat('./MatrixA/A%s'%M_no)
    M = data1['M']
    A1=M
    A=torch.from_numpy(A1).float()
    #At=torch.from_numpy(np.transpose(A1))
    loss_func=nn.MSELoss()
    ss=0
    #gradient descent
    for step, (x,b_label) in enumerate(test_loader):
        b_x = x.view(-1, 128 * 128) #original image
        y = torch.mm(A,torch.t(b_x)).detach()
        #################
        ## ADD NOISE
        #################
        if Noise_exist == True:
            data2 = sio.loadmat('./Noise-10dB/%s/Noise%s' % (cc, step + 1))
            Noise = torch.from_numpy(data2['Noise']).float()
            test_y = y + Noise / 255
        else:
            test_y = y
        s_hat=torch.mm(torch.t(A),test_y)
        s_hat=torch.t(s_hat)
        mse_err_p=1e6
        mse_step=0
        umin=0.7 #PGD training step
        while (True):
            decoded = autoencoder(s_hat)
            x_hat = decoded.view(128 * 128, 1)
            #calculate the gradient
            tem = test_y - torch.mm(A, x_hat)
            gradient=torch.mm(torch.t(A),tem)
            #decide if we need to stop
            compare=x_hat+umin*gradient
            compare=torch.t(compare)
            mse_err=loss_func(test_y,torch.mm(A,torch.t(compare)).detach())
            if mse_err>mse_err_p:
                output=x_hat.view(-1,128*128)
                break
            mse_err_p=mse_err
            s_hat=compare
        #calculate PSNR
        MSE=loss_func(output,b_x)
        xxx=torch.max(b_x)**2/MSE
        PSNR=10*torch.log10(xxx)
        sumPSNR[cc]=PSNR+sumPSNR[cc]
        if Noise_exist == True:
            np.savetxt('./Reconstructed Images/NN1-PGD-10dB/%s/imageoutput%s.txt'%(cc,step),output.detach().numpy())
        else:
            np.savetxt('./Reconstructed Images/NN1-PGD/%s/imageoutput%s.txt' % (cc, step), output.detach().numpy())
        np.savetxt('./Reconstructed Images/Original_Images_data/imageorigin%s.txt'%step,b_x.detach().numpy())
    print('PSNR:',sumPSNR[cc]/test_images)
