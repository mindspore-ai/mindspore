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
from pyDOE import lhs


class PINNs_training_set:
    """
    Training set for PINNs (Schrodinger)

    Args:
        N0 (int): number of sampled training data points for the initial condition
        Nb (int): number of sampled training data points for the boundary condition
        Nf (int): number of sampled training data points for the collocation points
        lb (np.array): lower bound (x, t) of domain
        ub (np.array): upper bound (x, t) of domain
        path (str): path of dataset
    """
    def __init__(self, N0, Nb, Nf, lb, ub, path='./Data/NLS.mat'):
        data = scio.loadmat(path)
        self.N0 = N0
        self.Nb = Nb
        self.Nf = Nf
        self.lb = lb
        self.ub = ub

        # load data
        t = data['tt'].flatten()[:, None]
        x = data['x'].flatten()[:, None]
        Exact = data['uu']
        Exact_u = np.real(Exact)
        Exact_v = np.imag(Exact)

        idx_x = np.random.choice(x.shape[0], self.N0, replace=False)
        self.x0 = x[idx_x, :]
        self.u0 = Exact_u[idx_x, 0:1]
        self.v0 = Exact_v[idx_x, 0:1]

        idx_t = np.random.choice(t.shape[0], self.Nb, replace=False)
        self.tb = t[idx_t, :]

        self.X_f = self.lb + (self.ub-self.lb)*lhs(2, self.Nf)

    def __getitem__(self, index):

        if index < self.N0:  # N0 initial points
            x = np.array([self.x0[index][0]], np.float32)
            t = np.array([0], np.float32)
            u_target = np.array(self.u0[index], np.float32)
            v_target = np.array(self.v0[index], np.float32)

        elif self.N0 <= index < self.N0+self.Nb: # Nb lower bound points
            ind = index - self.N0
            x = np.array([self.lb[0]], np.float32)
            t = np.array([self.tb[ind][0]], np.float32)
            u_target = np.array([self.ub[0]], np.float32)
            v_target = t

        elif self.N0+self.Nb <= index < self.N0+2*self.Nb: # Nb upper bound points
            ind = index - self.N0 - self.Nb
            x = np.array([self.ub[0]], np.float32)
            t = np.array([self.tb[ind][0]], np.float32)
            u_target = np.array([self.lb[0]], np.float32)
            v_target = t

        else: # Nf collocation points
            ind = index - self.N0 - 2*self.Nb
            x = np.array(self.X_f[ind, 0:1], np.float32)
            t = np.array(self.X_f[ind, 1:2], np.float32)
            u_target = np.array([0], np.float32)
            v_target = np.array([0], np.float32)

        return np.hstack((x, t)), np.hstack((u_target, v_target))

    def __len__(self):
        return self.N0+2*self.Nb+self.Nf


def generate_PINNs_training_set(N0, Nb, Nf, lb, ub, path='./Data/NLS.mat'):
    """
    Generate training set for PINNs

    Args: see class PINNs_train_set
    """
    s = PINNs_training_set(N0, Nb, Nf, lb, ub, path)
    dataset = ds.GeneratorDataset(source=s, column_names=['data', 'label'], shuffle=False)
    dataset = dataset.batch(batch_size=len(s))
    return dataset


def get_eval_data(path):
    """
    Get the evaluation data for Schrodinger equation.
    """
    data = scio.loadmat(path)
    t = data['tt'].astype(np.float32).flatten()[:, None]
    x = data['x'].astype(np.float32).flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact).astype(np.float32)
    Exact_v = np.imag(Exact).astype(np.float32)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    return X_star, u_star, v_star, h_star
