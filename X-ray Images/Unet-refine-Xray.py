import torch
import torch.nn as nn
import numpy as np
import os

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
autoencoder=UNet(in_ch=1,out_ch=1).eval()
autoencoder.load_state_dict(torch.load('Unet-refine-Xray.pkl', map_location=lambda storage, loc:storage)) #load the parameter values of NN
imagesize = 128
n=imagesize*imagesize #picture size
m=np.array([np.int(n*0.005),np.int(n*0.01),np.int(n*0.03), np.int(n*0.05),np.int(n*0.1),np.int(n*0.2)])

for file in range(0,len(m)): #for 4 different m
    j=0
    for name in os.listdir(r'./Reconstructed Images/4NNs-PGD/%s' % file):
        output = np.loadtxt('./Reconstructed Images/4NNs-PGD/%s/imageoutput%s.txt' % (file, j))
        output1 = torch.from_numpy(output).view(128,128).float()
        output1 = output1.unsqueeze(0)
        output1 = output1.unsqueeze(0)
        refine = autoencoder(output1).view(-1,128*128)
        np.savetxt('./Reconstructed Images/Unet-refine/%s/imageoutput%s.txt' % (file, j), refine.detach().numpy())
        j = j + 1