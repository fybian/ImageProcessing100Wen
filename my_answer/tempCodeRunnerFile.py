    img = RGB2GRAY(img)
    img[img < 128]=0
    img[img >= 128] = 255
    return img