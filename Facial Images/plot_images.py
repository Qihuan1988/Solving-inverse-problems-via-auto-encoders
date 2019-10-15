import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

f, a = plt.subplots(4, 3, figsize=(15, 20))
image_number=22

i=0
def set_yaxes(i,j,ylabel):
    a[i][j].axes.get_xaxis().set_visible(False)
    a[i][j].axes.get_yaxis().set_ticks([])
    a[i][j].set_ylabel('%s'%ylabel, fontsize=30)

for i in range(0,3):
    ii = i+1
    output1 = np.loadtxt('./Reconstructed Images/NN-PGD/%s/imageoutput%s.txt' % (ii,image_number))
    origin = np.loadtxt('./Reconstructed Images/Original_Images_data/%s/imageoutput%s.txt' % (ii,image_number))
    output2 = np.loadtxt('./Reconstructed Images/Unet-refine/%s/imageoutput%s.txt' % (ii,image_number))
    data1=sio.loadmat('./Reconstructed Images/BM3D-AMP/%s/im%s'%(ii,image_number+1))
    data=data1['x_hatx']/255
    originx=np.zeros([64,64,3])
    output1x = np.zeros([64, 64, 3])
    output2x = np.zeros([64, 64, 3])
    origin = np.reshape(origin, (3, 64, 64))
    output1 = np.reshape(output1, (3, 64, 64))
    output2 = np.reshape(output2, (3, 64, 64))
    for jj in range(0,3):
        originx[:,:,jj]=origin[jj,:,:]
        output1x[:,:,jj]=output1[jj,:,:]
        output2x[:,:,jj]=output2[jj,:,:]
    a[0][i].imshow(originx)
    if i==0:
        set_yaxes(0,0,'Original')
    else:
        a[0][i].axis('off')
    a[1][i].imshow(output1x)
    if i == 0:
        set_yaxes(1, 0, 'NN-PGD')
    else:
        a[1][i].axis('off')
    a[2][i].imshow(output2x)
    if i==0:
        set_yaxes(2,0,'Refine')
    else:
        a[2][i].axis('off')
    #a[4][i].imshow(np.reshape(output4[i, :], (128, 128)), cmap='gray')
    #a[4][i].axis('off')
    a[3][i].imshow(np.reshape(data,  (64, 64,3)))
    a[3][i].axes.get_xaxis().set_ticks([])
    a[3][i].axes.get_yaxis().set_ticks([])
    if i==0:
        a[3][i].set_ylabel('BM3D-AMP', fontsize=30)
        a[3][i].set_xlabel('0.05', fontsize=30)
    if i==1:
        a[3][i].set_xlabel('0.1', fontsize=30)
    if i==2:
        a[3][i].set_xlabel('0.2', fontsize=30)
    #if i==3:
     #   a[3][i].set_xlabel('0.2', fontsize=30)
plt.savefig('image%s.png'%image_number)