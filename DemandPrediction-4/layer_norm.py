# encoding utf-8
'''
@Author: william
@Description:
@time:2020/6/29 19:17
'''
import torch

# def layer_norm1(x,):
#     '''
#     Layer normalization function.
#     :param x: tensor, [batch_size, time_step, n_route, channel].
#     :param scope: str, variable scope.
#     :return: tensor, [batch_size, time_step, n_route, channel].
#     '''
#     _, N, C = x.get_shape().as_list()
#     mu, sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
#
#     with tf.variable_scope(str(scope)):
#         gamma = tf.get_variable('gamma', initializer=tf.ones([1, N, C]))
#         beta = tf.get_variable('beta', initializer=tf.zeros([1, N, C]))
#         _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
#     return _x


def layer_normal(x):
    mean, sigma = torch.mean(x), torch.var(x)
    x_ = (x - mean) / torch.sqrt(sigma)
    return x_