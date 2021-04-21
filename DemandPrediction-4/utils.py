import os
import zipfile
import numpy as np
import torch
import pandas as pd



def load_metr_la_data():
    # if (not os.path.isfile("data/adj_mat.npy")
    #         or not os.path.isfile("data/node_values.npy")):
    #     with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
    #         zip_ref.extractall("data/")

    # X = np.load("data/node_values_origin.npy")
    # X = X.transpose((1, 2, 0))
    # A = np.load("data/adj_mat_origin.npy")
    A=np.array(pd.read_csv("./data/201611-W_matrix.csv",header=None).values)
    X=np.array(pd.read_csv("./data/201611final-V_matrix.csv",header=None).values)
    # A = np.load("./data/201611-W_matrix.csv")
    # X = np.load("./data/201611final-V_matrix.csv")       #2小时
    # X = np.load("data_process/V_matrix90t1.0.npy")         #1小时
    # X = np.load("data_process/V_matrix90t.npy")  # 1个半小时

    # Weather = np.load("data/Weather_matrix576.npy").astype(np.float32)

    # X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))
    #X = np.reshape(X, (X.shape[0], 24, 24))
    # X = np.reshape(X, (1365, 24, 24))

    ##X_val = np.load("data/V_matrix576_test.npy")        #2小时100个验证
    X_val=X
    # X_val = np.load("data/V_matrix90t_test1.0.npy")          #1小时100个验证
    # X_val = np.load("data/V_matrix90t_test.npy")          # 1个半小时100个验证
    # X_val = np.load("data/node_val/es.npy")
    # X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1)).transpose((1, 2, 0))
    #X_val = np.reshape(X_val, (X_val.shape[0], 24, 24))
    # Weather_val = np.load("data/Weather_matrix576_test.npy").astype(np.float32)

    A = A.astype(np.float32)
    # A = A.astype(np.float64)
    X = X.astype(np.float32)
    X_val = X_val.astype(np.float32)
    # X = X.astype(np.float64)
    # X_val = X_val.astype(np.float64)
    # Normalization using Z-score method
    means = np.mean(X)
    stds = np.std(X)
    #标准化
    X = X - means.reshape(1, -1, 1)     #-1固定，其他变为1
    X_val = X_val - means.reshape(1, -1, 1)
    X = X / stds.reshape(1, -1, 1)
    X_val = X_val / stds.reshape(1, -1, 1)

    return A, X.transpose((1, 2, 0)), means, stds, X_val.transpose((1, 2, 0))
    # mean_weather = np.mean(Weather)
    # std_weather = np.std(Weather)
    # Weather = (Weather - mean_weather) / std_weather

    # Weather_val = ((Weather_val - mean_weather) / std_weather)

    # max_weather = np.max(Weather)
    # min_weather = np.min(Weather)
    # Weather = (Weather - max_weather) / (max_weather - min_weather)

    # return A, X, means, stds, X_val, Weather, Weather_val

    # Normalization using Max-Min method
    # max_X, min_X = np.max(X), np.min(X)
    # _range = max_X - min_X
    # X = (X - min_X) / _range
    # return A, X, max_X, min_X

    # Normalization using Max method
    # max_value = X.max()
    # X = X / max_value
    # X_val = X_val / max_value
    # return A, X, max_value, X_val

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points

    indices = [[int(j) for j in range(i - (num_timesteps_input*29), i+29, 29)] for i
               in range(num_timesteps_input*29, X.shape[0])]

    # Save samples
    features, target = [], []
    for i in indices:
        features.append(X[i[:-1], :, :].transpose(
                (0, 2, 1)))
        target.append(X[i[-1:], :, :].transpose(
                (0, 2, 1)))

    return torch.squeeze(torch.from_numpy(np.array(features)),dim=2), \
           torch.squeeze(torch.from_numpy(np.array(target)),dim=2)


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of meadard deviation.
    :return: np.ndarray, z-score normalin.
    :param std: float, the value of stanzed array.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    temp_data = np.mean(np.abs(v_ - v) / (v + 1e-5))
    return temp_data


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))

