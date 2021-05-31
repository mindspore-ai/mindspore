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
"""Eval"""
import argparse

import numpy as np
from mindspore import Tensor, context
from mindspore import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype

from src import config
from src.Schrodinger.dataset import get_eval_data
from src.Schrodinger.net import PINNs

def eval_PINNs_sch(ckpoint_name, num_neuron=100, path='./Data/NLS.mat'):
    """
    Evaluation of PINNs for Schrodinger equation scenario.

    Args:
        ckpoint_name (str): model checkpoint file name
        num_neuron (int): number of neurons for fully connected layer in the network
        path (str): path of the dataset for Schrodinger equation
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    layers = [2, num_neuron, num_neuron, num_neuron, num_neuron, 2]
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    n = PINNs(layers, lb, ub)
    param_dict = load_checkpoint(ckpoint_name)
    load_param_into_net(n, param_dict)

    X_star, _, _, h_star = get_eval_data(path)

    X_tensor = Tensor(X_star, mstype.float32)
    pred = n(X_tensor)
    u_pred = pred[0].asnumpy()
    v_pred = pred[1].asnumpy()
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    error_h = np.linalg.norm(h_star-h_pred, 2)/np.linalg.norm(h_star, 2)
    print(f'evaluation error is: {error_h}')

    return error_h


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PINNs for Schrodinger equation scenario')
    parser.add_argument('--ckpoint_path', type=str, help='model checkpoint(ckpt) filename')

    #only support 'Schrodinger' for now
    parser.add_argument('--scenario', type=str, help='scenario for PINNs', default='Schrodinger')
    parser.add_argument('--datapath', type=str, help='path for dataset', default='')

    args_opt = parser.parse_args()
    f_name = args_opt.ck_file
    pinns_scenario = args_opt.scenario
    data_path = args_opt.datapath

    if pinns_scenario == 'Schrodinger':
        conf = config.config_Sch
        hidden_size = conf['num_neuron']
        if data_path == '':
            dataset_path = conf['path']
        else:
            dataset_path = data_path
        mse_error = eval_PINNs_sch(f_name, hidden_size, dataset_path)
    else:
        print(f'{pinns_scenario} is not supported in PINNs evaluation for now')
