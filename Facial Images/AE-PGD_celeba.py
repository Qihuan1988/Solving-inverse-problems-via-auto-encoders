import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms
import scipy.io as sio

#Load date from test data document
test_data = torchvision.datasets.ImageFolder('./Testing_set', transform=transforms.Compose([
    transforms.ToTensor()]))
test_loader=Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

#to load the trained neural network (same structures as the trained NN)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder=nn.Sequential( #two layers encoder
            nn.Linear(64*64*3,12000),
            nn.Sigmoid(), #ReLU, Tanh, etc.
            nn.Linear(12000,3000),
            nn.Sigmoid(), #input is in (0,1), so select this one
        )
        self.decoder=nn.Sequential( #two layers decoder
            nn.Linear(3000,12000),
            nn.Sigmoid(),
            nn.Linear(12000, 64*64*3),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded

#Load the trained NN parameters to the new NN
autoencoder=Autoencoder()
autoencoder.load_state_dict(torch.load('AE-trained.pkl', map_location=lambda storage, loc:storage)) #load the parameter values of NN

n=64*64*3 #picture size
m=np.array([np.int(n*0.01),np.int(n*0.05),np.int(n*0.1), np.int(n*0.2)]) #compressed image size
test_images=100
sumPSNR=np.zeros(len(m)) #total PSNR

for cc in range(0,len(m)): #for different m values
    #get the matrix A
    M_no=cc+1
    data1 = sio.loadmat('./MatrixA/A%s'%M_no)
    M = data1['M']
    A1=M
    A=torch.from_numpy(A1).float()
    loss_func=nn.MSELoss()
    #gradient descent
    for step, (x,b_label) in enumerate(test_loader):
        b_x = x.view(-1, 64 * 64*3) #original image
        #image data for each channel
        x1 = x[0,0,:,:]
        x2 = x[0,1,:,:]
        x3 = x[0,2,:,:]
        b_x1 = x1.view(-1, 64 * 64)
        b_x2 = x2.view(-1, 64 * 64)
        b_x3 = x3.view(-1, 64 * 64)
        y1 = torch.mm(A,torch.t(b_x1)).detach()
        y2 = torch.mm(A, torch.t(b_x2)).detach()
        y3 = torch.mm(A, torch.t(b_x3)).detach()
        y = torch.cat([y1, y2, y3], dim=1)
        s_hat1=torch.mm(torch.t(A),y1)
        s_hat2 = torch.mm(torch.t(A), y2)
        s_hat3 = torch.mm(torch.t(A), y3)
        s_hat=torch.t(torch.cat([s_hat1,s_hat2,s_hat3],dim=0))
        mse_err_p=1e16
        if cc==0:
            umin=0.2 #GD training step
        if cc==1:
            umin=0.5
        if cc==2:
            umin=0.7
        if cc==3:
            umin=0.9
        while (True):
            decoded = autoencoder(s_hat)
            x_hat = decoded.view(64 * 64 * 3, 1)
            x_hat1 = torch.narrow(x_hat , 0, 0, 64*64)
            x_hat2 = torch.narrow(x_hat, 0, 64 * 64, 64 * 64)
            x_hat3 = torch.narrow(x_hat, 0, 64 * 64 * 2, 64 * 64)
            #calculate the gradient
            tem1 = y1 - torch.mm(A, x_hat1)
            tem2 = y2 - torch.mm(A, x_hat2)
            tem3 = y3 - torch.mm(A, x_hat3)
            gradient1 = torch.mm(torch.t(A), tem1)
            gradient2 = torch.mm(torch.t(A), tem2)
            gradient3 = torch.mm(torch.t(A), tem3)
            compare1 = torch.t(x_hat1 + umin * gradient1)
            compare2 = torch.t(x_hat2 + umin * gradient2)
            compare3 = torch.t(x_hat3 + umin * gradient3)
            compare = torch.cat([compare1, compare2, compare3], dim=1)
            mse_err=loss_func(torch.cat([y1,y2,y3], dim=1),torch.cat([torch.mm(A,torch.t(compare1)).detach(),
                                                                      torch.mm(A,torch.t(compare2)).detach(),
                                                                      torch.mm(A,torch.t(compare2)).detach()],dim=1))
            # decide if we need to stop
            if mse_err>mse_err_p:
                break
            output = x_hat.view(-1, 64 * 64 * 3)
            mse_err_p=mse_err
            s_hat=compare
        #calculate PSNR
        MSE=loss_func(output,b_x)
        xxx=torch.max(b_x)**2/MSE
        PSNR=10*torch.log10(xxx)
        sumPSNR[cc]=PSNR+sumPSNR[cc]
        #storing the output images and original images
        #np.savetxt('./Reconstructed Images/NN-PGD/%s/imageoutput%s.txt'%(cc,step),output.detach().numpy())
        #np.savetxt('./Reconstructed Images/Original_Images_data/%s/orignalimage%s.txt'%(cc,step),b_x.detach().numpy())
    print('PSNR:',sumPSNR[cc]/test_images)


