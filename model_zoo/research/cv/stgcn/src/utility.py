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
"""
Calculate laplacian matrix, used to network weight.
Evaluate the performance of net work.
"""

import numpy as np
import mindspore.ops as ops

from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs

def calculate_laplacian_matrix(adj_mat, mat_type):
    """
    calculate laplacian matrix used for graph convolution layer.
    """
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    #deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    # Combinatorial
    com_lap_mat = deg_mat - adj_mat

    # For SpectraConv
    # To [0, 1]
    sym_normd_lap_mat = np.matmul(np.matmul(fractional_matrix_power(deg_mat, -0.5), \
     com_lap_mat), fractional_matrix_power(deg_mat, -0.5))

    # For ChebConv
    # From [0, 1] to [-1, 1]
    lambda_max_sym = eigs(sym_normd_lap_mat, k=1, which='LR')[0][0].real
    wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / lambda_max_sym - id_mat

    # For GCNConv
    wid_deg_mat = deg_mat + id_mat
    wid_adj_mat = adj_mat + id_mat
    hat_sym_normd_lap_mat = np.matmul(np.matmul(fractional_matrix_power(wid_deg_mat, -0.5), \
     wid_adj_mat), fractional_matrix_power(wid_deg_mat, -0.5))

    # Random Walk
    rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)

    # For SpectraConv
    # To [0, 1]
    rw_normd_lap_mat = id_mat - rw_lap_mat

    # For ChebConv
    # From [0, 1] to [-1, 1]
    lambda_max_rw = eigs(rw_lap_mat, k=1, which='LR')[0][0].real
    wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat

    # For GCNConv
    wid_deg_mat = deg_mat + id_mat
    wid_adj_mat = adj_mat + id_mat
    hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)

    if mat_type == 'wid_sym_normd_lap_mat':
        return wid_sym_normd_lap_mat
    if mat_type == 'hat_sym_normd_lap_mat':
        return hat_sym_normd_lap_mat
    if mat_type == 'wid_rw_normd_lap_mat':
        return wid_rw_normd_lap_mat
    if mat_type == 'hat_rw_normd_lap_mat':
        return hat_rw_normd_lap_mat
    raise ValueError(f'ERROR: "{mat_type}" is unknown.')

def evaluate_metric(model, dataset, scaler):
    """
    evaluate the performance of network.
    """
    mae, sum_y, mape, mse = [], [], [], []
    for data in dataset.create_dict_iterator():
        x = data['inputs']
        y = data['labels']
        y_pred = model(x)
        y_pred = ops.Reshape()(y_pred, (len(y_pred), -1))
        y_pred = scaler.inverse_transform(y_pred.asnumpy()).reshape(-1)
        y = scaler.inverse_transform(y.asnumpy()).reshape(-1)
        d = np.abs(y - y_pred)
        mae += d.tolist()
        sum_y += y.tolist()
        mape += (d / y).tolist()
        mse += (d ** 2).tolist()
    MAE = np.array(mae).mean()
    MAPE = np.array(mape).mean()
    RMSE = np.sqrt(np.array(mse).mean())
    #WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

    return MAE, RMSE, MAPE
