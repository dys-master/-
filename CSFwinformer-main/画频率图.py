import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图像
# img = cv2.imread(r'D:\study\remote_sensing\mmsegmentation\am100836_sat.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread(r'/home/louis/SemSeg/cuda11.3mmseg/mmsegmentation-master/zyb_work/mass/全层输出/解码/hwc-448-448-48解码4—mlp.jpg', 1)####[H,W,C]//cv2  bgr

img = cv2.imread(r'D:\Code\CSFwinformer-main\data\1500\1500\images\train\1.png', 1)####[H,W,C]//cv2  bgr

img=cv2.resize(img, dsize=(512,512))
b,g,r=cv2.split(img)
img=cv2.merge([r,g,b]).transpose(2,0,1)

# 进行离散傅里叶变换
dft =np.fft.fft2(img, axes=(1,2))
dft_shift = np.fft.fftshift(dft)


# 进行傅立叶逆变换并显示结果


A=cv2.normalize((np.angle(dft_shift)), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U).transpose(1,2,0)
A[:,:,0]=A[:,:,0]*0.4
A[:,:,1]=A[:,:,1]
A[:,:,2]=255-0.5*A[:,:,2]

plt.imsave('解码4_ori_ang.png',A,cmap='viridis')

plt.subplot(423)
plt.imshow(cv2.normalize(np.log(np.abs(dft_shift)), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U).transpose(1,2,0))
plt.title('ori_mag')
plt.axis("off")

A=cv2.normalize(np.log(np.abs(dft_shift)), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U).transpose(1,2,0)
A[:,:,0]=A[:,:,0]*0.4
A[:,:,1]=A[:,:,1]
A[:,:,2]=255-0.5*A[:,:,2]

plt.imsave('解码4_ori_mag.png',A,cmap='viridis')