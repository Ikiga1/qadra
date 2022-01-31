import matplotlib.pyplot as plt
from zipfile import ZipFile
import os, sys, wget
import numpy as np
import cv2
import glob

def download_dataset():
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    tiny_imgdataset = wget.download('http://cs231n.stanford.edu/tiny-imagenet-200.zip', out = os.getcwd())
    for file in os.listdir(os.getcwd()):
        if file.endswith(".zip"):
            zip = ZipFile(file)
            zip.extractall()
        else:
            print("not found")

#https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f
# Functions to display single or a batch of sample images
def imshow(img):
    plt.imshow(img)
    plt.show()

def load_img(img_path, size):
    img = cv2.imread(img_path)
    # make the image black and white
    # https://holypython.com/python-pil-tutorial/how-to-convert-an-image-to-black-white-in-python-pil/
    x, y, _ = img.shape
    img_bw = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            img_bw[i][j] = img[i][j][0]*0.2125+img[i][j][1]*0.7174+img[i][j][2]*0.0721

    img_bw = cv2.resize(img_bw,(size,size), interpolation = cv2.INTER_CUBIC)
    return img_bw

def load_dataset(image_paths, size):
    n = len(image_paths)
    m = size*size
    dataset = np.zeros((n, m))
    for i in range(n):
        img = load_img(image_paths[i], size)
        dataset[i] = img.reshape(m)
    return dataset
