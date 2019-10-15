import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

mn = [0,1,2,3]
psnr_bm3d = np.zeros(4)
psnr_bm3d_std = np.zeros(4)
psnr_4AEs = np.zeros(4)
psnr_4AEs_std = np.zeros(4)
psnr_refine = np.zeros(4)
psnr_refine_std = np.zeros(4)

for file in range(4):
    psnr_bm3d_each = np.zeros(100)
    psnr_4AEs_each = np.zeros(100)
    psnr_refine_each = np.zeros(100)
    i = 0
    j = 0
    k = 0
    for name in os.listdir(r'./Reconstructed Images/BM3D-AMP/%s'%file):
        i = i + 1
        data1 = sio.loadmat('./Reconstructed Images/BM3D-AMP/%s/im%s' % (file,i))
        output = data1['x_hatx']
        output = np.reshape(output,(1,64*64*3))
        orig1 = sio.loadmat('./Reconstructed Images/Original_Images_data(matlab)/orignalimage%s' % i)
        origin =  orig1['ImIn']
        origin = np.reshape(origin,(1,64*64*3))
        mse = np.sum((output-origin)**2)/(64*64*3)
        psnr_bm3d_each[i-1] = 10*np.log10(np.max(origin)**2/mse)
    for name in os.listdir(r'./Reconstructed Images/NN-PGD/%s'%file):
        output1 = np.loadtxt('./Reconstructed Images/NN-PGD/%s/imageoutput%s.txt' % (file, j))
        origin1 = np.loadtxt('./Reconstructed Images/Original_Images_data/orignalimage%s.txt'%j)
        mse = np.sum((output1-origin1)**2)/(64*64*3)
        psnr_4AEs_each[j] = 10*np.log10(np.max(origin1)**2/mse)
        j = j + 1
    for name in os.listdir(r'./Reconstructed Images/Unet-refine/%s'%file):
        output2 = np.loadtxt('./Reconstructed Images/Unet-refine/%s/imageoutput%s.txt' % (file, k))
        origin2 = np.loadtxt('./Reconstructed Images/Original_Images_data/orignalimage%s.txt'%k)
        mse = np.sum((output2-origin2)**2)/(64*64*3)
        psnr_refine_each[k] = 10*np.log10(np.max(origin2)**2/mse)
        k = k + 1

    psnr_bm3d[file] = np.mean(psnr_bm3d_each)
    psnr_bm3d_std[file] = np.std(psnr_bm3d_each)
    psnr_4AEs[file] = np.mean(psnr_4AEs_each)
    psnr_4AEs_std[file] = np.std(psnr_4AEs_each)
    psnr_refine[file] = np.mean(psnr_refine_each)
    psnr_refine_std[file] = np.std(psnr_refine_each)
plt.errorbar(mn,psnr_4AEs,psnr_4AEs_std,fmt='-*',ecolor='orangered',color='r',elinewidth=2,capsize=4)
#plt.errorbar(mn,psnr_refine,psnr_refine_std,fmt='-o',ecolor='deepskyblue',color='b',elinewidth=2,capsize=4)
plt.errorbar(mn,psnr_bm3d,psnr_bm3d_std,fmt='-^',ecolor='palegreen',color='g',elinewidth=2,capsize=4)

plt.xlabel('Sample Rate(m/n)')
plt.ylabel('Average PSNR(dB)')
plt.xticks(mn,('0.01','0.05','0.1','0.2'))
plt.legend(['NN-PGD','BM3D-AMP'])
plt.savefig('psnr.png')