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

from src import config
from src.NavierStokes.eval_ns import eval_PINNs_navier
from src.Schrodinger.eval_sch import eval_PINNs_sch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PINNs for Schrodinger equation scenario')
    parser.add_argument('--ckpoint_path', type=str, help='model checkpoint(ckpt) filename')

    parser.add_argument('--scenario', type=str, help='scenario for PINNs', default='Schrodinger')
    parser.add_argument('--datapath', type=str, help='path for dataset', default='')

    args_opt = parser.parse_args()
    f_name = args_opt.ckpoint_path
    pinns_scenario = args_opt.scenario
    data_path = args_opt.datapath

    if pinns_scenario in ['Schrodinger', 'Sch', 'sch', 'quantum']:
        conf = config.config_Sch
        hidden_size = conf['num_neuron']
        if data_path == '':
            dataset_path = conf['path']
        else:
            dataset_path = data_path
        mse_error = eval_PINNs_sch(f_name, hidden_size, dataset_path)
    elif pinns_scenario in ['ns', 'NavierStokes', 'navier', 'Navier']:
        conf = config.config_navier
        hidden_size = conf['num_neuron']
        if data_path == '':
            dataset_path = conf['path']
        else:
            dataset_path = data_path
        error = eval_PINNs_navier(f_name, dataset_path, hidden_size)
    else:
        print(f'{pinns_scenario} is not supported in PINNs evaluation for now')
