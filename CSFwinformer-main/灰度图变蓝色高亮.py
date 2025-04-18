import cv2
import numpy as np
import matplotlib.pyplot as plt





# 读取图像并转换为灰度图像
# img = cv2.imread(r'D:\study\remote_sensing\mmsegmentation\am100836_sat.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread(r'/home/louis/SemSeg/cuda11.3mmseg/mmsegmentation-master/zyb_work/mass/全层输出/解码/hwc-448-448-48解码4—mlp.jpg', 1)####[H,W,C]//cv2  bgr

img = cv2.imread(r'D:\Code\CSFwinformer-main\data\1500\1500\images\train\10.png', 1)####[H,W,C]//cv2  bgr
b,g,r=cv2.split(img)
img=cv2.merge([r,g,b]).transpose(2,0,1)
A=cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U).transpose(1,2,0)
A[:,:,0]=A[:,:,0]*0.4
A[:,:,1]=A[:,:,1]
A[:,:,2]=255-0.5*A[:,:,2]
plt.imsave('hwc-448-448-48解码4—mlp.jpg',A,cmap='viridis')