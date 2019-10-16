import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

f, a = plt.subplots(4, 5, figsize=(20, 20))
image_number=50

i=0
def set_yaxes(i,j,ylabel):
    a[i][j].axes.get_xaxis().set_visible(False)
    a[i][j].axes.get_yaxis().set_ticks([])
    a[i][j].set_ylabel('%s'%ylabel, fontsize=30)

for i in range(0,5):
    output1 = np.loadtxt('./Reconstructed Images/NN-PGD/%s/imageoutput%s.txt' % (i+1,image_number))
    origin = np.loadtxt('./Reconstructed Images/Original_Images_data/imageorigin%s.txt' %image_number)
    output2 = np.loadtxt('./Reconstructed Images/LASSO-DCT/%s/imageoutput%s.txt' % (i+1,image_number))
    data1=sio.loadmat('./Reconstructed Images/BM3D-AMP/%s/im%s'%(i+1,image_number+1))
    data=data1['x_hat']
    a[0][i].imshow(np.reshape(origin, (28, 28)), cmap='gray')
    if i==0:
        set_yaxes(0,0,'Original')
    else:
        a[0][i].axis('off')
    a[1][i].imshow(np.reshape(output1, (28, 28)), cmap='gray')
    if i == 0:
        set_yaxes(1, 0, 'NN-PGD')
    else:
        a[1][i].axis('off')
    a[2][i].imshow(np.reshape(output2, (28, 28)), cmap='gray')
    if i == 0:
        set_yaxes(2, 0, 'LASSO-DCT')
    else:
        a[2][i].axis('off')
    a[3][i].imshow(np.reshape(data, (28, 28)), cmap='gray')
    a[3][i].axes.get_xaxis().set_ticks([])
    a[3][i].axes.get_yaxis().set_ticks([])
    if i==0:
        a[3][i].set_ylabel('BM3D-AMP', fontsize=30)
        a[3][i].set_xlabel('0.05', fontsize=30)
    if i==1:
        a[3][i].set_xlabel('0.1', fontsize=30)
    if i==2:
        a[3][i].set_xlabel('0.2', fontsize=30)
    if i==3:
        a[3][i].set_xlabel('0.3', fontsize=30)
    if i==4:
        a[3][i].set_xlabel('0.4', fontsize=30)
plt.savefig('image%s.png'%image_number)