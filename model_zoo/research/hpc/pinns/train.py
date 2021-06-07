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
"""Train PINNs"""
import argparse

from src import config
from src.NavierStokes.train_ns import train_navier
from src.Schrodinger.train_sch import train_sch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PINNs')

    parser.add_argument('--scenario', type=str, help='scenario for PINNs', default='Schrodinger')
    parser.add_argument('--datapath', type=str, help='path for dataset', default='')
    parser.add_argument('--noise', type=float, help='noise, Navier-Stokes only', default='-0.5')
    parser.add_argument('--epoch', type=int, help='number of epochs for training', default=0)

    args_opt = parser.parse_args()
    pinns_scenario = args_opt.scenario
    data_path = args_opt.datapath
    epoch_num = args_opt.epoch

    if pinns_scenario in ['Schrodinger', 'Sch', 'sch', 'quantum']:
        conf = config.config_Sch
        if data_path != '':
            conf['path'] = data_path
        if epoch_num > 0:
            conf['epoch'] = epoch_num
        train_sch(**conf)
    elif pinns_scenario in ['ns', 'NavierStokes', 'navier', 'Navier']:
        conf = config.config_navier
        if data_path != '':
            conf['path'] = data_path
        noise = args_opt.noise
        if noise >= 0:
            conf['noise'] = noise
        if epoch_num > 0:
            conf['epoch'] = epoch_num
        train_navier(**conf)
    else:
        print(f'{pinns_scenario} is not supported in PINNs training for now')
