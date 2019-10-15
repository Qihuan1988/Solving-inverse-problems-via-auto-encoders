import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import torchvision.transforms as transforms
import scipy.io as sio
import scipy.fftpack as fftpack
import copy

test_data = torchvision.datasets.ImageFolder('./Testing_set', transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
test_loader=Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

n=128*128 #picture size
m=np.array([np.int(n*0.005),np.int(n*0.01),np.int(n*0.03), np.int(n*0.05),np.int(n*0.1),np.int(n*0.2)]) #compressed picture size
test_images=100
sumPSNR=np.zeros(len(m))
#matrix A needs to do DCT
def dctA(vector):
    size=128
    image2d=np.reshape(vector,[size,size])
    dctimage = fftpack.dct(fftpack.dct(image2d,axis=0,norm='ortho'),axis=1,norm='ortho')
    dctvector = dctimage.reshape([-1])
    return dctvector
#The reconstructed image needs to do IDCT
def idctOutput(vector):
    size = 128
    image2d = np.reshape(vector, [size, size])
    idctimage = fftpack.idct(fftpack.idct(image2d,axis=0, norm='ortho'),axis=1, norm='ortho')
    idctvector = idctimage.reshape([-1])
    return idctvector

Noise_exist=False
if Noise_exist==True:
    start_num=3
else:
    start_num=0
for cc in range(start_num,len(m)):
    #gradient descent
    M_no = cc + 1
    data1 = sio.loadmat('./MatrixA/A%s' % M_no)
    M = data1['M']
    A = M
    A_new = copy.deepcopy(A)
    #do DCT to matrix A
    for i in range(0,m[cc]):
        A_new[i,:]=dctA(A[i,:])
    for step, (x,b_label) in enumerate(test_loader):
        b_xx = x.view(-1, 128 * 128) #stand for x in the paper
        b_x = Variable(b_xx).data.numpy()
        y = np.dot(A,np.transpose(b_x))
        if Noise_exist == True:
            data2 = sio.loadmat('./Noise-10dB/%s/Noise%s' % (cc, step + 1))
            Noise = torch.from_numpy(data2['Noise']).float()
            test_y = y + Noise / 255
        else:
            test_y = y
        model = Lasso(alpha=0.00001)
        model.fit(A_new,test_y.ravel())
        predicted = model.coef_.reshape(1,128*128)
        idctpredicted=idctOutput(predicted).reshape(1,128*128)
        #calculate PSNR
        MSE=mean_squared_error(b_x,idctpredicted)
        xxx=np.max(b_x)**2/MSE
        PSNR=10*np.log10(xxx)
        sumPSNR[cc]=PSNR+sumPSNR[cc]
        if Noise_exist == True:
            np.savetxt('./Reconstructed Images/LASSO-DCT-10dB/%s/imageoutput%s.txt' % (cc, step), output.detach().numpy())
        else:
            np.savetxt('./Reconstructed Images/LASSO-DCT/%s/imageoutput%s.txt' % (cc, step), output.detach().numpy())
    print('PSNR:', sumPSNR[cc]/test_images)
