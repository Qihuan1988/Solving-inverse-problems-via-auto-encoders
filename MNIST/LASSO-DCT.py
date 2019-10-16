import torch
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import scipy.io as sio

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
)
#Data loader
test_loader=Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
n=28*28 #picture size
m=np.array([np.int(n*0.05),np.int(n*0.1),np.int(n*0.2), np.int(n*0.3),np.int(n*0.4)]) #compressed picture size
test_images=300
#Projected gradient descent
sumPSNR=np.zeros(len(m))
Noise_exist=False
if Noise_exist==True:
    start_num=1
else:
    start_num=0
for cc in range(start_num,len(m)): #for 4 different m
    M_no = cc + 1
    data1 = sio.loadmat('./Re_image/MATRIX/A_mnist%s' % M_no)
    M = data1['M']
    A1 = M
    A = torch.from_numpy(A1).float()
    At = torch.from_numpy(np.transpose(A1))
    #gradient descent
    for step, (x,b_label) in enumerate(test_loader):
        b_xx = x.view(-1, 28 * 28) #stand for x in the paper
        b_x = Variable(b_xx).data.numpy()
        y = np.dot(A,np.transpose(b_x)).ravel()
        if Noise_exist == True:
            data2 = sio.loadmat('./Noise-10dB/%s/Noise%s' % (M_no, step + 1))
            Noise = torch.from_numpy(data2['Noise']).float()
            test_y = y + Noise / 255
        else:
            test_y = y
        test_y = np.transpose(test_y)
        #model = Lasso(alpha=0.001)
        model = Lasso(alpha=0.00002)
        model.fit(A,test_y)
        predicted = model.coef_.reshape(1,784)
        #calculate PSNR
        MSE=mean_squared_error(b_x,predicted)
        xxx=np.max(b_x)**2/MSE
        PSNR=10*np.log10(xxx)
        sumPSNR[cc]=PSNR+sumPSNR[cc]
        if Noise_exist == True:
            np.savetxt('./Reconstructed Images/LASSO-DCT-10dB/%s/imageoutput%s.txt' % (M_no, step), predicted)
        else:
            np.savetxt('./Reconstructed Images/LASSO-DCT/%s/imageoutput%s.txt' % (M_no, step), predicted)
        if step==test_images-1:
            break
    print('PSNR:',sumPSNR/test_images)