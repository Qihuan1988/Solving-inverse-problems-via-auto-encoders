import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

mn = [0,1,2,3,4,5]
#NN capacity values are got from NNcapacity.py
high = [28.5,28.5,28.5,28.5,28.5,28.5]
low = [26.2,26.2,26.2,26.2,26.2,26.2]

psnr_bm3d = np.zeros(6)
psnr_bm3d_std = np.zeros(6)
psnr_low = np.zeros(6)
psnr_low_std = np.zeros(6)
psnr_high = np.zeros(6)
psnr_high_std = np.zeros(6)
psnr_lasso = np.zeros(6)
psnr_lasso_std = np.zeros(6)

for file in range(6):
    psnr_bm3d_each = np.zeros(100)
    psnr_low_each = np.zeros(100)
    psnr_high_each = np.zeros(100)
    psnr_lasso_each = np.zeros(100)
    i = 0
    j = 0
    k = 0
    l = 0
    for name in os.listdir(r'./Reconstructed Images/BM3D-AMP/%s'%file):
        i = i + 1
        data1 = sio.loadmat('./Reconstructed Images/BM3D-AMP/%s/im%s' % (file,i))
        output = data1['x_hat']
        output = np.reshape(output,(1,128*128))
        data2 = sio.loadmat('./Reconstructed Images/Original_Images_data(matlab)/Origin%s.mat'%i)
        origin = data2['ImIn']
        origin = np.reshape(origin, (1, 128 * 128))
        mse = np.sum((output-origin)**2)/128**2
        psnr_bm3d_each[i-1] = 10*np.log10(np.max(origin)**2/mse)
    for name in os.listdir(r'./Reconstructed Images/NN1-PGD/%s'%file):
        output1 = np.loadtxt('./Reconstructed Images/NN1-PGD/%s/imageoutput%s.txt' % (file, j))
        origin1 = np.loadtxt('./Reconstructed Images/Original_Images_data/imageorigin%s.txt'%j)
        mse = np.sum((output1-origin1)**2)/128**2
        psnr_low_each[j] = 10*np.log10(np.max(origin1)**2/mse)
        j = j + 1
    for name in os.listdir(r'./Reconstructed Images/NN2-PGD/%s'%file):
        output2 = np.loadtxt('./Reconstructed Images/NN2-PGD/%s/imageoutput%s.txt' % (file, k))
        origin2 = np.loadtxt('./Reconstructed Images/Original_Images_data/imageorigin%s.txt'%k)
        mse = np.sum((output2-origin2)**2)/128**2
        psnr_high_each[k] = 10*np.log10(np.max(origin2)**2/mse)
        k = k + 1
    for name in os.listdir(r'./Reconstructed Images/LASSO-DCT/%s'%file):
        output3 = np.loadtxt('./Reconstructed Images/LASSO-DCT/%s/imageoutput%s.txt' % (file, l))
        origin3 = np.loadtxt('./Reconstructed Images/Original_Images_data/imageorigin%s.txt'%l)
        mse = np.sum((output3-origin3)**2)/128**2
        psnr_lasso_each[l] = 10*np.log10(np.max(origin2)**2/mse)
        l = l + 1
    psnr_bm3d[file] = np.mean(psnr_bm3d_each)
    psnr_bm3d_std[file] = np.std(psnr_bm3d_each)
    psnr_low[file] = np.mean(psnr_low_each)
    psnr_low_std[file] = np.std(psnr_low_each)
    psnr_high[file] = np.mean(psnr_high_each)
    psnr_high_std[file] = np.std(psnr_high_each)
    psnr_lasso[file] = np.mean(psnr_lasso_each)
    psnr_lasso_std[file] = np.std(psnr_lasso_each)
plt.errorbar(mn,psnr_low,psnr_low_std,fmt='-*',ecolor='orangered',color='r',elinewidth=2,capsize=4)
plt.plot(mn,low,'crimson')
plt.errorbar(mn,psnr_high,psnr_high_std,fmt='-o',ecolor='deepskyblue',color='b',elinewidth=2,capsize=4)
plt.plot(mn,high,'cornflowerblue')
plt.errorbar(mn,psnr_bm3d,psnr_bm3d_std,fmt='-^',ecolor='palegreen',color='g',elinewidth=2,capsize=4)
plt.errorbar(mn,psnr_lasso,psnr_lasso_std,fmt='-v',ecolor='orange',color='y',elinewidth=2,capsize=4)

plt.xlabel('Sample Rate(m/n)')
plt.ylabel('Average PSNR(dB)')
plt.xticks(mn,('0.005','0.01','0.03','0.05','0.1','0.2'))
plt.legend(['NN1 Capacity','NN2 Capacity','NN1-PGD','NN2-PGD','BM3D-AMP','Lasso-DCT'])
plt.savefig('psnr1.png')