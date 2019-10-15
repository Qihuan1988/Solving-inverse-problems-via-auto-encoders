import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import os
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"]='0'
ids=[0]
torch.cuda.empty_cache()
#initial values
EPOCH = 20 #this can give a good result
BATCH_SIZE = 64
LR = 0.001

# load images data
train_data = torchvision.datasets.ImageFolder('./Training_set', transform=transforms.Compose([
transforms.ToTensor()]))
train_loader=Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class UNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

unet=UNet(in_ch=3,out_ch=3).cuda()
optimizer=torch.optim.Adam(unet.parameters(),lr=LR) #optimizer: SGD, Momentum, Adagrad, etc. This one is better.
loss_func=nn.MSELoss() #loss function: MSE

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
        return decoded

#Load 4 NNs
autoencoder=Autoencoder().cuda()
autoencoder.load_state_dict(torch.load('AE-trained.pkl')) #load the parameter values of NN

for epoch in range(EPOCH):
    for step, (x,x_label) in enumerate(train_loader): #train_loader has the number of batches, data, and label
        b_y=x.cuda()
        batch_size = x.size()[0]
        b_x=x.view(-1,64*64*3).cuda()
        decoded=autoencoder(b_x)
        decoded=decoded.view(batch_size,3,64,64)
        decoded=decoded.cuda()
        #running in the neural network
        output=unet(decoded)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()#initialize the optimizer
        loss.backward()
        optimizer.step()

        if step%40==0:
            print('Epoch:',epoch, '| tran loss : %.4f' % loss.data.cpu().numpy())

torch.save(unet.state_dict(),'Unet-trained.pkl') #save the parameter values of neural network
