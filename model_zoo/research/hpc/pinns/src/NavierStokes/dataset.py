# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Create dataset for training or evaluation"""
import mindspore.dataset as ds
import numpy as np
import scipy.io as scio


class data_set_navier_stokes:
    """
    Training set for PINNs(Navier-Stokes)

    Args:
        n_train (int): amount of training data
        path (str): path of dataset
        noise (float): noise intensity, 0 for noiseless training data
        train (bool): True for training set, False for evaluation set
    """
    def __init__(self, n_train, path, noise, train=True):
        data = scio.loadmat(path)
        self.n_train = n_train
        self.noise = noise

        # load data
        X_star = data['X_star'].astype(np.float32)
        t_star = data['t'].astype(np.float32)
        U_star = data['U_star'].astype(np.float32)

        N = X_star.shape[0]  # number of data points per time step
        T = t_star.shape[0]  # number of time steps

        XX = np.tile(X_star[:, 0:1], (1, T))
        YY = np.tile(X_star[:, 1:2], (1, T))
        TT = np.tile(t_star, (1, N)).T
        UU = U_star[:, 0, :]
        VV = U_star[:, 1, :]

        x = XX.flatten()[:, None]
        y = YY.flatten()[:, None]
        t = TT.flatten()[:, None]
        u = UU.flatten()[:, None]
        v = VV.flatten()[:, None]

        self.lb = np.array([np.min(x), np.min(y), np.min(t)], np.float32)
        self.ub = np.array([np.max(x), np.max(y), np.max(t)], np.float32)

        if train:
            idx = np.random.choice(N*T, n_train, replace=False)  # sampled data points
            self.noise = noise
            self.x = x[idx, :]
            self.y = y[idx, :]
            self.t = t[idx, :]
            u_train = u[idx, :]
            self.u = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
            v_train = v[idx, :]
            self.v = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])
        else:
            self.x = x
            self.y = y
            self.t = t
            self.u = u
            self.v = v

            P_star = data['p_star'].astype(np.float32)
            PP = P_star
            self.p = PP.flatten()[:, None]

    def __getitem__(self, index):
        ans_x = self.x[index]
        ans_y = self.y[index]
        ans_t = self.t[index]
        ans_u = self.u[index]
        ans_v = self.v[index]
        input_data = np.hstack((ans_x, ans_y, ans_t)).astype(np.float32)
        label = np.hstack((ans_u, ans_v, np.array([0.]))).astype(np.float32)  #
        return input_data, label

    def __len__(self):
        return self.n_train


def generate_training_set_navier_stokes(batch_size, n_train, path, noise):
    """
    Generate training set for PINNs (Navier-Stokes)

    Args:
        batch_size (int): amount of training data per batch
        n_train (int): amount of training data
        path (str): path of dataset
        noise (float): noise intensity, 0 for noiseless training data
    """
    s = data_set_navier_stokes(n_train, path, noise, True)
    lb = s.lb
    ub = s.ub
    dataset = ds.GeneratorDataset(source=s, column_names=['data', 'label'], shuffle=True)
    dataset = dataset.batch(batch_size)
    return dataset, lb, ub
