import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms

#Load date from test data document
test_data = torchvision.datasets.ImageFolder('./Testing_set', transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
test_loader=Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

#to load the trained neural network (same structures as the trained NN)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder=nn.Sequential( #two layers encoder
            nn.Linear(128*128,8000),
            nn.ReLU(True), #ReLU, Tanh, etc.
            nn.Linear(8000,3000),
            nn.ReLU(True) #input is in (0,1), so select this one
        )
        self.decoder=nn.Sequential( #two layers decoder
            nn.Linear(3000, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 128*128),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded

#Load the trained NN parameters to the new NN
autoencoder=Autoencoder()
autoencoder.load_state_dict(torch.load('NN2-Xray.pkl', map_location=lambda storage, loc:storage)) #load the parameter values of NN

n=128*128 #picture size
m=np.array([np.int(n*0.005),np.int(n*0.01),np.int(n*0.03), np.int(n*0.05),np.int(n*0.1),np.int(n*0.2)]) #compressed picture size
test_images=100
sumPSNR=np.zeros(len(m)) #total PSNR
loss_func=nn.MSELoss()
for cc in range(0,len(m)): #for different m values
    for step, (x,b_label) in enumerate(test_loader):
        b_x = x.view(-1, 128 * 128) #original image
        while (True):
            decoded = autoencoder(b_x)
            output = decoded.view(-1,128*128)
            break
        MSE=loss_func(output,b_x)
        xxx=torch.max(b_x)**2/MSE
        PSNR=10*torch.log10(xxx)
        sumPSNR[cc]=PSNR+sumPSNR[cc]
    print('PSNR:', sumPSNR[cc] / test_images)
