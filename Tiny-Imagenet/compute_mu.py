import numpy as np
import sys
import random
import json
import gc

data = np.load('./tiny_imagenet_bw.npy')

data_mean0 = np.array(data - np.mean(data, axis = 0))

random.seed(1234)
index_list = [x for x in range(data.shape[0])]
random.shuffle(index_list)

def __mu(p, matrix):
    def s(p, A):
        if p == 0:
            result = np.max([np.count_nonzero(A[i]) for i in range(len(A))])
        else:
            result = np.max([np.sum(np.power(abs(A[i]), [p]*len(A[i]))) for i in range(len(A))])#np.max([pow(np.linalg.norm(A[i], ord=p), p) for i in range(len(A))])
        gc.collect()
        return result

    s1 = s(2 * p, matrix)
    s2 = s(2 * (1 - p), matrix.T)
    mu = np.sqrt(s1 * s2)

    gc.collect()
    return mu

def linear_search(matrix, start, step):
    domain = [i for i in np.arange(start, 1.0, step)]
    values = [__mu(i, matrix) for i in domain]
    domain = domain
    best_p = domain[values.index(min(values))]
    return best_p, min(values)

def best_mu(matrix, start, step):
    p, val = linear_search(matrix, start, step)
    val_list = [val, np.linalg.norm(matrix)]
    index = val_list.index(min(val_list))
    if index == 0:
        best_norm = f"p={p}"
    elif index == 1:
        best_norm = "Frobenius"        

    return best_norm, val_list[index]

with open("result_mu.json") as f:
	info_dict = json.load(f)

size = int(sys.argv[1])
print(size)
info_dict['sizes'].append(size)
X_mean0 = data_mean0[index_list[:size]]

_, E, _ = np.linalg.svd(X_mean0, full_matrices=False)

p, mu = best_mu(X_mean0, 0.0001, 0.05)
info_dict['nonnorm_ps'].append(p)
info_dict['nonnorm_mu'].append(mu)

A = X_mean0/E[0]

p, mu = best_mu(A, 0.0001, 0.05)
info_dict['norm_ps'].append(p)
info_dict['norm_mu'].append(mu)

with open('result_mu.json', 'w') as fp:
	json.dump(info_dict, fp)