# encoding utf-8
'''
@Author: william
@Description:
@time:2020/10/27 20:49
'''

import numpy as np

x = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4],
    [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6]
]

x = np.array(x)
x = np.reshape(x, (x.shape[0], 4, 4))
means = np.mean(x, axis=(0, 2))
print(x)