from loadingutils import *
import numpy as np
import glob
import os
from collections import Counter
import random
import math
import gc

data = np.load('./tiny_imagenet_bw.npy')

data_mean0 = np.array(data - np.mean(data, axis = 0))

random.seed(1234)
index_list = [x for x in range(data.shape[0])]
random.shuffle(index_list)

spectral_norms = []

print("Ready!")

for size in range(1000, data.shape[0]+1, 1000):
	print(size)
	X_mean0 = data_mean0[index_list[:size]]

	_, E, _ = np.linalg.svd(X_mean0, full_matrices=False)
	spectral_norms.append(E[0])

	del X_mean0
	del E
	gc.collect()

info_dict = {'sizes':[size for size in range(1000, data.shape[0]+1, 1000)],
'spectral_norms': spectral_norms}

import json
with open('result_spectral.json', 'w') as fp:
	json.dump(info_dict, fp)