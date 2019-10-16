import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

mn = [0,1,2,3]
psnr_bm3d = np.zeros(4)
psnr_bm3d_std = np.zeros(4)
psnr_AE = np.zeros(4)
psnr_AE_std = np.zeros(4)
psnr_lasso = np.zeros(4)
psnr_lasso_std = np.zeros(4)

for file in range(2,6):
    psnr_bm3d_each = np.zeros(300)
    psnr_AE_each = np.zeros(300)
    psnr_lasso_each = np.zeros(300)
    i = 0
    j = 0
    k = 0
    for name in os.listdir(r'./Reconstructed Images/BM3D-AMP-10dB/%s'%file):
        i = i + 1
        ori = i-1
        data1 = sio.loadmat('./Reconstructed Images/BM3D-AMP-10dB/%s/im%s' % (file,i))
        output = data1['x_hat']
        output = np.reshape(output,(28*28))
        origin = np.loadtxt('./Reconstructed Images/Original_Images_data/imageorigin%s.txt'%ori)*255
        origin = np.reshape(origin,(28*28))
        mse = np.sum((output-origin)**2)/28**2
        psnr_bm3d_each[ori] = 10*np.log10(np.max(origin)**2/mse)
    for name in os.listdir(r'./Reconstructed Images/NN-PGD-10dB/%s'%file):
        output1 = np.loadtxt('./Reconstructed Images/NN-PGD-10dB/%s/imageoutput%s.txt' % (file, j))
        origin1 = np.loadtxt('./Reconstructed Images/Original_Images_data/imageorigin%s.txt'%j)
        mse = np.sum((output1-origin1)**2)/28**2
        psnr_AE_each[j] = 10*np.log10(np.max(origin1)**2/mse)
        j = j + 1
    for name in os.listdir(r'./Reconstructed Images/LASSO-DCT-10dB/%s'%file):
        output2 = np.loadtxt('./Reconstructed Images/LASSO-DCT-10dB/%s/imageoutput%s.txt' % (file, k))
        origin2 = np.loadtxt('./Reconstructed Images/Original_Images_data/imageorigin%s.txt'%k)
        mse = np.sum((output2-origin2)**2)/28**2
        psnr_lasso_each[k] = 10*np.log10(np.max(origin2)**2/mse)
        k = k + 1

    psnr_bm3d[file-2] = np.mean(psnr_bm3d_each)
    psnr_bm3d_std[file-2] = np.std(psnr_bm3d_each)
    psnr_AE[file-2] = np.mean(psnr_AE_each)
    psnr_AE_std[file-2] = np.std(psnr_AE_each)
    psnr_lasso[file-2] = np.mean(psnr_lasso_each)
    psnr_lasso_std[file-2] = np.std(psnr_lasso_each)
plt.errorbar(mn,psnr_AE,psnr_AE_std,fmt='-*',ecolor='orangered',color='r',elinewidth=2,capsize=4)
plt.errorbar(mn,psnr_lasso,psnr_lasso_std,fmt='-o',ecolor='deepskyblue',color='b',elinewidth=2,capsize=4)
plt.errorbar(mn,psnr_bm3d,psnr_bm3d_std,fmt='-^',ecolor='palegreen',color='g',elinewidth=2,capsize=4)

plt.xlabel('Sample Rate(m/n)')
plt.ylabel('Average PSNR(dB)')
plt.xticks(mn,('0.1','0.2','0.3','0.4'))
plt.legend(['AutoEncoder','LASSO','BM3D-AMP'])
plt.savefig('psnrn.png')