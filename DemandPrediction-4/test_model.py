# encoding utf-8
'''
@Author: william
@Description:
@time:2020/7/1 10:25
'''

import torch
from data_load import Data_load
from stgcn import STGCN
from utils import get_normalized_adj

if __name__ == '__main__':
    num_timesteps_input = 12
    num_timesteps_output = 9

    net = STGCN(25, 1, num_timesteps_input, num_timesteps_output).cuda()
    net = torch.load('./checkpoints/params_400.pkl')

    A, means, stds, training_input, training_target, val_input, val_target, test_input, test_target = Data_load(
        num_timesteps_input, num_timesteps_output)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    val_input = val_input.cuda()

    out = net(A_wave, val_input)

    print('')