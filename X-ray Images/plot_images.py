import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

f, a = plt.subplots(5, 6, figsize=(20, 20))
image_number=23

i=0
def set_yaxes(i,j,ylabel):
    a[i][j].axes.get_xaxis().set_visible(False)
    a[i][j].axes.get_yaxis().set_ticks([])
    a[i][j].set_ylabel('%s'%ylabel, fontsize=30)

for i in range(0,6):
    output1 = np.loadtxt('./Reconstructed Images/NN1-PGD/%s/imageoutput%s.txt' % (i,image_number))
    origin = np.loadtxt('./Reconstructed Images/Original_Images_data/imageorigin%s.txt' % image_number)
    output2 = np.loadtxt('./Reconstructed Images/NN2-PGD/%s/imageoutput%s.txt' % (i,image_number))
    output3 = np.loadtxt('./Reconstructed Images/LASSO-DCT/%s/imageoutput%s.txt' % (i,image_number))
    data1=sio.loadmat('./Reconstructed Images/BM3D-AMP/%s/im%s'%(i,image_number+1))
    data=data1['x_hat']
    a[0][i].imshow(np.reshape(origin, (128, 128)), cmap='gray')
    if i==0:
        set_yaxes(0,0,'Original')
    else:
        a[0][i].axis('off')
    a[1][i].imshow(np.reshape(output1, (128, 128)), cmap='gray')
    if i == 0:
        set_yaxes(1, 0, 'NN1-PGD')
    else:
        a[1][i].axis('off')
    a[2][i].imshow(np.reshape(output2, (128, 128)), cmap='gray')
    if i==0:
        set_yaxes(2,0,'NN2-PGD')
    else:
        a[2][i].axis('off')
    a[3][i].imshow(np.reshape(output3, (128, 128)), cmap='gray')
    if i == 0:
        set_yaxes(3, 0, 'LASSO-DCT')
    else:
        a[3][i].axis('off')
    #a[4][i].imshow(np.reshape(output4[i, :], (128, 128)), cmap='gray')
    #a[4][i].axis('off')
    a[4][i].imshow(np.reshape(data, (128, 128)), cmap='gray')
    a[4][i].axes.get_xaxis().set_ticks([])
    a[4][i].axes.get_yaxis().set_ticks([])
    if i==0:
        a[4][i].set_ylabel('BM3D-AMP', fontsize=30)
        a[4][i].set_xlabel('0.005', fontsize=30)
    if i==1:
        a[4][i].set_xlabel('0.01', fontsize=30)
    if i==2:
        a[4][i].set_xlabel('0.03', fontsize=30)
    if i==3:
        a[4][i].set_xlabel('0.05', fontsize=30)
    if i==4:
        a[4][i].set_xlabel('0.1', fontsize=30)
    if i==5:
        a[4][i].set_xlabel('0.2', fontsize=30)
plt.savefig('image%s.png'%image_number)