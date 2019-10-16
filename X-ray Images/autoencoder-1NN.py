import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import os
from torchvision import transforms

#initial values
EPOCH = 50
BATCH_SIZE = 64
LR = 0.0005
N_TEST_IMG = 5
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
ids = [0]
torch.cuda.empty_cache()
# load images data
train_data = torchvision.datasets.ImageFolder('./Training_set', transform=transforms.Compose([
    transforms.ToTensor(),]))
train_loader=Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

#NN network
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
        return encoded, decoded

autoencoder=Autoencoder().cuda()
optimizer=torch.optim.Adam(autoencoder.parameters(),lr=LR) #optimizer: SGD, Momentum, Adagrad, etc. This one is better.
loss_func=nn.MSELoss() #loss function: MSE

for epoch in range(EPOCH):
    for step, (x,x_label) in enumerate(train_loader): #train_loader has the number of batches, data, and label
        b_x=x.view(-1,128*128).cuda() #input data
        b_y=x.view(-1,128*128).cuda() #comparing data
        #running in the neural network
        encoded, decoded=autoencoder(b_x)
        loss=loss_func(decoded,b_y)
        optimizer.zero_grad()#initialize the optimizer
        loss.backward()
        optimizer.step()

        if step%10==0:
            print('Epoch:',epoch, '| tran loss : %.4f' % loss.data.cpu().numpy(),'count:', len(b_x[1,:]))
torch.save(autoencoder.state_dict(),'NN2-Xray.pkl') #save the parameter values of neural network
