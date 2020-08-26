import cv2
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读入原始图像
img_orign = cv2.imread("Question_01_10/imori.jpg")
img_noise = cv2.imread("Question_01_10/imori_noise.jpg")
img_dark = cv2.imread("Question_11_20/imori_dark.jpg")
# 交换通道
def bgr2rgb(img):
    img[:, :, (0,1,2)] = img[:, :, (2,1,0)]
    return img

# 灰度化
def rgb2gray(img):
    blue  = img[:, :, 0].copy()
    green = img[:, :, 1].copy()
    red   = img[:, :, 2].copy()
    gray = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    gray = gray.astype(np.uint8)
    return gray

# 二值化
def rgb_binarization(img, th=128): 
    # 转灰度图   
    img = RGB2GRAY(img)
    img[img < th] = 0
    img[img >= th] = 255
    return img

# 大津二值化算法
def rgb_otsu_binarization(img):
    # 转灰度图
    img = RGB2GRAY(img)
    H,W = img.shape
    max_th = 0
    max_Sb = 0
    # 寻找最佳阈值
    for th in range(100, 256):
        v0 = img[np.where(img < th)]
        w0 = len(v0) / (H * W) 
        m0 = np.mean(v0) if len(v0) > 0 else 0
        # print(w0, m0)
        v1 = img[np.where(img > th)]
        w1 = len(v1) / (H * W) 
        m1 = np.mean(v1) if len(v1) > 0 else 0
        # print(w1, m1)
        Sb = w0 * w1 * ((m0 -m1) ** 2)
        # print(Sb)
        if Sb > max_Sb:
            max_Sb = Sb
            max_th = th
    # 二值化
    img[img < max_th] = 0
    img[img >= max_th] = 255

    return img

# HSV变换
def rgb2hsv():
    return 0

# 减色处理
def color_quantization(img):
    out = img.copy()
    out = out // 64 * 64 + 32
    return out

# 平均池化
def averge_pooling(img, G=8):
    out = img.copy()
    H, W, C = out.shape
    NH = H // G
    NW = W // G
    for i in range(NH):
        for j in range(NW):
            for k in range(C):
                out[i*G:(i+1)*G, j*G:(j+1)*G, k] = np.mean(out[i*G:(i+1)*G, j*G:(j+1)*G, k]).astype(np.int)
    return out

# 最大池化
def max_pooling(img, G=8):
    out = img.copy()
    H, W, C = out.shape
    NH = H // G
    NW = W // G
    for i in range(NH):
        for j in range(NW):
            for k in range(C):
                out[i*G:(i+1)*G, j*G:(j+1)*G, k] = np.amax(out[i*G:(i+1)*G, j*G:(j+1)*G, k]).astype(np.int)
    return out

# 高斯滤波
def gaussian_filter(img, ker_size=3, sigma=1.3):

    H, W, C = img.shape
   
    pad = ker_size // 2

    # 计算滤波器边界
    if ker_size % 2 != 0:
        upper_limit = pad + 1
    else:
        upper_limit = pad 

    # 计算滤波器参数
    ker = np.zeros((ker_size,ker_size))
    for i in range(-pad, upper_limit):
        for j in range(-pad, upper_limit):
            ker[i+pad, j+pad] = np.exp(-(i**2 + j**2)/(2 * sigma**2))
    ker /= (sigma * np.sqrt(2 * np.pi))    
    ker /= ker.sum()

    # 滤波
    tmp = img.copy()
    # 输出数组
    out = np.zeros((H-ker_size+1, W-ker_size+1, C), dtype=np.float)
    
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
            for k in range(C):
                out[i, j, k] = np.sum(ker * tmp[i:i+ker_size, j:j+ker_size, k])

    out = out.astype(np.uint8)

    return out

# 中值滤波
def median_filter(img, ker_size=3):
    H, W, C = img.shape
    tmp = img.copy()
    out = np.zeros((H-ker_size+1, W-ker_size+1, C), dtype=np.float)
    
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
            for k in range(C):
                out[i, j, k] = np.median(tmp[i:i+ker_size, j:j+ker_size, k])
    out = out.astype(np.uint8)
    
    return out

# 均值滤波
def mean_filter(img, ker_size=3):
    H, W, C = img.shape
    tmp = img.copy()
    out = np.zeros((H-ker_size+1, W-ker_size+1, C), dtype=np.float)
    
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
            for k in range(C):
                out[i, j, k] = np.mean(tmp[i:i+ker_size, j:j+ker_size, k])
    out = out.astype(np.uint8)
    
    return out

# 动态模糊
def motion_filter(img, ker_size=3):
    H, W, C = img.shape
    tmp = img.copy()
    out = np.zeros((H-ker_size+1, W-ker_size+1, C), dtype=np.float)
    
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
            for k in range(C):
                out[i, j, k] = np.mean([tmp[i, j, k], tmp[i+1, j+1, k], tmp[i+2, j+2, k]])
    out = out.astype(np.uint8)
    
    return out    

# max-min 滤波器
def max_min_filter(img, ker_size=3):

    tmp = rgb2gray(img)
    H, W = tmp.shape
    out = np.zeros((H-ker_size+1, W-ker_size+1), dtype=np.float)
    
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
                out[i, j] = np.max(tmp[i:i+ker_size, j:j+ker_size])-np.min(tmp[i:i+ker_size, j:j+ker_size])
    out = out.astype(np.uint8)
    
    return out  

# 微分滤波器
def differential_filter(img, select=1, ker_size=3):
    tmp = rgb2gray(img)
    H, W = tmp.shape
    ker_h = [[0., -1., 0.], [0., 1., 0.], [0., 0., 0.]]
    ker_v = [[0., 0., 0.], [-1., 1., 0.], [0., 0., 0.]]
    # select=1 纵向 select=2 横向
    if select == 1:
        ker = ker_h
    elif select == 2:    
        ker = ker_v
        
    out = np.zeros((H-ker_size+1, W-ker_size+1), dtype=np.float)
    
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
                out[i, j] = np.sum(ker * tmp[i:i+ker_size, j:j+ker_size])
 
    out[out < 0] = 0
    out[out > 255] = 255

    out = out.astype(np.uint8)
   
    return out

# Sobel滤波器
def sobel_filter(img, select=1, ker_size=3):
    tmp = rgb2gray(img)
    H, W = tmp.shape
    ker_h = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]
    ker_v = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
    # select=1 纵向 select=2 横向
    if select == 1:
        ker = ker_h
    elif select == 2:    
        ker = ker_v
        
    out = np.zeros((H-ker_size+1, W-ker_size+1), dtype=np.float)
    
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
                out[i, j] = np.sum(ker * tmp[i:i+ker_size, j:j+ker_size])
 
    out[out < 0] = 0
    out[out > 255] = 255

    out = out.astype(np.uint8)
   
    return out

# prewitt滤波器
def prewitt_filter(img, select=1, ker_size=3):
    tmp = rgb2gray(img)
    H, W = tmp.shape
    ker_h = [[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]
    ker_v = [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]
    # select=1 纵向 select=2 横向
    if select == 1:
        ker = ker_h
    elif select == 2:    
        ker = ker_v
        
    out = np.zeros((H-ker_size+1, W-ker_size+1), dtype=np.float)
    
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
                out[i, j] = np.sum(ker * tmp[i:i+ker_size, j:j+ker_size])
 
    out[out < 0] = 0
    out[out > 255] = 255

    out = out.astype(np.uint8)
   
    return out

# laplacian滤波器
def laplacian_filter(img, select=1, ker_size=3):
    tmp = rgb2gray(img)
    H, W = tmp.shape
    ker = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
        
    out = np.zeros((H-ker_size+1, W-ker_size+1), dtype=np.float)
    
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
                out[i, j] = np.sum(ker * tmp[i:i+ker_size, j:j+ker_size])
 
    out[out < 0] = 0
    out[out > 255] = 255

    out = out.astype(np.uint8)
   
    return out

# Emboss滤波器
def emboss_filter(img, select=1, ker_size=3):
    tmp = rgb2gray(img)
    H, W = tmp.shape
    ker = [[-2., -1., 0.], [-1., 1., 1.], [0., 1., 2.]]
        
    out = np.zeros((H-ker_size+1, W-ker_size+1), dtype=np.float)
    
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
                out[i, j] = np.sum(ker * tmp[i:i+ker_size, j:j+ker_size])
 
    out[out < 0] = 0
    out[out > 255] = 255

    out = out.astype(np.uint8)
   
    return out

# LoG滤波器#####有问题
def LoG_filter(img, ker_size=3, sigma=1.3):
    tmp = rgb2gray(img)
    H, W = tmp.shape
    pad = ker_size // 2
    # 计算滤波器边界
    if ker_size % 2 != 0:
        upper_limit = pad + 1
    else:
        upper_limit = pad 
    # 计算滤波器参数
    ker = np.zeros((ker_size,ker_size))
    for i in range(-pad, upper_limit):
        for j in range(-pad, upper_limit):
            ker[i+pad, j+pad] = (i**2 + j**2 -sigma**2) * np.exp(-(i**2 + j**2)/(2 * sigma**2))
    ker /= (2 * np.pi * sigma**6)    
    ker /= ker.sum()
    # 输出数组
    out = np.zeros((H-ker_size+1, W-ker_size+1), dtype=np.float)
    for i in range(H-ker_size+1):
        for j in range(W-ker_size+1):
                out[i, j] = np.sum(ker * tmp[i:i+ker_size, j:j+ker_size])
    out = out.astype(np.uint8)
    return out


# img_processed = LoG_filter(img_noise,ker_size=3)
# cv2.imshow("orign", img_orign)
# cv2.imshow("noise", img_noise)
# cv2.imshow("dark", img_dark)
# cv2.imshow("process", img_processed)
# cv2.waitKey(0)
plt.hist(img_dark.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out.png")
plt.show()