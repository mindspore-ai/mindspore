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
"""export checkpoint file into mindir models"""
import argparse

from src import config
from src.Schrodinger.export_sch import export_sch
from src.NavierStokes.export_ns import export_ns


parser = argparse.ArgumentParser(description='PINNs export')
parser.add_argument('--ckpoint_path', type=str, help='model checkpoint(ckpt) filename')
parser.add_argument('--file_name', type=str, help='export file name')
parser.add_argument('--scenario', type=str, help='scenario for PINNs', default='Schrodinger')
parser.add_argument('--datapath', type=str, help='path for dataset', default='')
parser.add_argument('--batch_size', type=int, help='batch size', default=0)


if __name__ == '__main__':
    args_opt = parser.parse_args()
    ck_file = args_opt.ckpoint_path
    file_format = 'MINDIR'
    file_name = args_opt.file_name
    pinns_scenario = args_opt.scenario
    dataset_path = args_opt.datapath
    b_size = args_opt.batch_size
    if pinns_scenario in ['Schrodinger', 'Sch', 'sch', 'quantum']:
        conf = config.config_Sch
        num_neuron = conf['num_neuron']
        N0 = conf['N0']
        Nb = conf['Nb']
        Nf = conf['Nf']
        export_sch(num_neuron, N0=N0, Nb=Nb, Nf=Nf, ck_file=ck_file,
                   export_format=file_format, export_name=file_name)
    elif pinns_scenario in ['ns', 'NavierStokes', 'navier', 'Navier']:
        conf = config.config_navier
        num_neuron = conf['num_neuron']
        if dataset_path != '':
            path = dataset_path
        else:
            path = conf['path']
        if b_size <= 0:
            batch_size = conf['batch_size']
        else:
            batch_size = b_size
        export_ns(num_neuron, path=path, ck_file=ck_file, batch_size=batch_size,
                  export_format=file_format, export_name=file_name)
    else:
        print(f'{pinns_scenario} scenario in PINNs is not supported to export for now')
