import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision
import numpy as np
import scipy.io as sio

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_loader=Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder=nn.Sequential( #two layers encoder
            nn.Linear(784,1500),
            nn.Sigmoid(), #ReLU, Tanh, etc.
            nn.Linear(1500,100),
            nn.Sigmoid(),#input is in (0,1), so select this one
        )
        self.decoder=nn.Sequential( #two layers decoder
            nn.Linear(100, 1500),
            nn.Sigmoid(),
            nn.Linear(1500, 784),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded

autoencoder=Autoencoder().eval()
autoencoder.load_state_dict(torch.load('Autoencoder(Sigmoid).pkl')) #load the parameter values of NN

n=28*28 #picture size
m=np.array([np.int(n*0.05),np.int(n*0.1),np.int(n*0.2), np.int(n*0.3),np.int(n*0.4)]) #compressed picture size
test_images=300
sumPSNR=np.zeros(len(m))
Noise_exist=False
if Noise_exist==True:
    start_num=1
else:
    start_num=0
for cc in range(start_num,len(m)):
    #get the matrix A
    M_no=cc+1
    data1 = sio.loadmat('./MatrixA/A_mnist%s'%M_no)
    M = data1['M']
    A1=M
    A=torch.from_numpy(A1).float()
    At=torch.from_numpy(np.transpose(A1))
    loss_func=nn.MSELoss()
    #gradient descent
    for step, (x,b_label) in enumerate(test_loader):
        b_x = x.view(-1, 28 * 28)
        y = torch.mm(A,torch.t(b_x)).detach()
        if Noise_exist == True:
            data2 = sio.loadmat('./Noise-10dB/%s/Noise%s' % (M_no, step + 1))
            Noise = torch.from_numpy(data2['Noise']).float()
            test_y = y + Noise / 255
        else:
            test_y = y
        s_hat=torch.mm(torch.t(A),test_y)
        s_hat=torch.t(s_hat) #stand for s in the paper
        mse_err_p=1e6
        mse_step=0
        umin=0.7
        vv=0
        while (True):
            decoded = autoencoder(s_hat)
            decodedtemp = decoded.view(28 * 28, 1)
            tem = test_y - torch.mm(A, decodedtemp)
            # gradient descent
            gradient = torch.mm(torch.t(A), tem)
            compare = decodedtemp + umin * gradient
            compare = torch.t(compare)
            mse_err = loss_func(test_y, torch.mm(A, torch.t(compare)))
            if mse_err > mse_err_p:
                vv=vv+1
                if vv==1:
                    output=decoded.view(-1,28*28)
                if vv==5:
                    break
                s_hat=compare
            else:
                s_hat = compare
                vv=0
                mse_err_p = mse_err
        #calculate PSNR
        MSE=loss_func(output,b_x)
        xxx=torch.max(b_x)**2/MSE
        PSNR=10*torch.log10(xxx)
        sumPSNR[cc]=PSNR+sumPSNR[cc]
        if Noise_exist == True:
            np.savetxt('./Reconstructed Images/NN-PGD-10dB/%s/imageoutput%s.txt'%(M_no,step),output.detach().numpy())
        else:
            np.savetxt('./Reconstructed Images/NN-PGD/%s/imageoutput%s.txt' % (M_no, step),
                       output.detach().numpy())
        np.savetxt('./Reconstructed Images/Original_Images_data/%s/imageorigin%s.txt'%(M_no,step),b_x.detach().numpy())
        if step==test_images-1:
            break
    print('PSNR:',sumPSNR[cc]/test_images)


