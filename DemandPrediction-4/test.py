# encoding utf-8
'''
@Author: william
@Description:
@time:2020/6/19 9:00
'''
import torch
import numpy as np

X = [[1, 2, 3, 4],
     [2, 2, 3, 4],
     [3, 2, 3, 4]
     ]
X = np.asarray(X)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))
# # means = np.mean(X, axis=(0, 2))
# # X = X - means.reshape(1, -1, 1)
# # stds = np.std(X, axis=(0, 2))
# # X = X / stds.reshape(1, -1, 1)
#
# _range = np.max(X) - np.min(X)
# X = (X - np.min(X)) / _range
X_ = torch.tensor(X)

print(X)