from loadingutils import *
import numpy as np
import glob
import os

##Check if imagenet is on disk, otherwise download it and extract it
if not os.path.isdir('./tiny-imagenet-200'):
    download_dataset()

images_paths = glob.glob('./tiny-imagenet-200/*/*/*/*')

# Select a subset of images to load
images_subset = images_paths[:]
# Select the size of the square images
size = 64
data = load_dataset(images_subset, size)

np.save("tiny_imagenet_bw.npy", data)