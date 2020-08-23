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
def Binarization(img):
    img = RGB2GRAY(img)
	img[img < 128] = 0
	img[img >= 128] = 255
    return img

img = Binarization(img)
cv2.imshow("imori", img)
cv2.waitKey(0)
