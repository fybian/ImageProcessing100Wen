import cv2
import numpy as np
# 读入图像
img = cv2.imread("Question_01_10/imori.jpg")

# 交换通道
def BGR2RGB(img):
    img[:, :, (0,1,2)] = img[:, :, (2,1,0)]
    return img

# 灰度化
def RGB2GRAY(img):
    blue = img[:, :, 0].copy()
    green = img[:, :, 1].copy()
    red = img[:, :, 2].copy()
    gray = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    gray = gray.astype(np.uint8)
    return gray

# 二值化
def RGB_Binarization(img, th=128): 
    # 转灰度图   
    img = RGB2GRAY(img)
    img[img < th] = 0
    img[img >= th] = 255
    return img

# 大津二值化算法
def RGB_otsuBinarization(img):
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
def RGB2HSV():
    return 0

# 减色处理
def Color_quantization(img):
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
def GaussianFilter(img):
    out = img.copy()

    return out

img_1 = GaussianFilter(img)
cv2.imshow("orign", img)
cv2.imshow("process", img_1)
cv2.waitKey(0)
