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

ks = []
smallest_gaps = []
epsilons_order = []
epsilons_threshold = []
linfty_norms = []
frobenius_norms = []
thetas = []

for size in range(1000, data.shape[0]+1, 1000):
	print(size)
	X_mean0 = data_mean0[index_list[:size]]

	U, E, V = np.linalg.svd(X_mean0, full_matrices=False)
	E2 = np.array([pow(e,2) for e in E]) # Computing the powers of 2
	E3 = E2/np.sum(E2) # Computing the Factor score ratios


	#### Distribution of the factor score ratios
	plt.plot(E3, color="darkblue")
	plt.grid()
	plt.xlabel("singular value index")
	plt.ylabel("factor score ratio")
	plt.savefig(f'TinyImagenet_sv_distribution_n{size}.pdf', bbox_inches='tight')

	real_exp_var = 0
	i=0
	while real_exp_var <= 0.85:
		real_exp_var += E3[i]
		i += 1
	k = i

	#### Compute epsilon for the low rank A with k=62
	#### epsilon has to be smaller than the smallest gap between (normalized) singular values
	NE = E/E[0]

	eps = 100
	index = 0
	for i in range(k):
		if NE[i] - NE[i+1] < eps:
			eps = NE[i] - NE[i+1]
		index = i

	ks.append(k)
	smallest_gaps.append(index)
	epsilons_order.append(eps)
	epsilons_threshold.append(NE[k-1]-NE[k])

	#### Computing other run-time parameters for matrix A
	A = X_mean0/E[0]

	index = 0
	max_ = 0
	for i in range(len(A)):
		l1_norm = np.sum([abs(a) for a in A[i]])
		if  l1_norm > max_:
			max_ = l1_norm
		index = i

	linfty_norms.append(max_)
	frobenius_norms.append(np.linalg.norm(A))
	thetas.append(NE[k-1])

	del X_mean0
	del U
	del E
	del V
	del NE
	del A
	gc.collect()

info_dict = {'sizes':[size for size in range(1000, data.shape[0]+1, 1000)],
'ks':ks,
'smallest_gaps':smallest_gaps,
'epsilon_order':epsilons_order,
'epsilon_threshold':epsilons_threshold,
'linfty_norms':linfty_norms,
'frobenius_norm':frobenius_norms,
'thetas':thetas}

import json
with open('result.json', 'w') as fp:
	json.dump(info_dict, fp)