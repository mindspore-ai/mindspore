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
"""export checkpoint file into air, onnx, mindir models"""
import argparse

import numpy as np
from mindspore import (Tensor, context, export, load_checkpoint,
                       load_param_into_net)
import mindspore.common.dtype as mstype

from src import config
from src.Schrodinger.net import PINNs

parser = argparse.ArgumentParser(description='PINNs export')
parser.add_argument('ck_file', type=str, help='model checkpoint(ckpt) filename')
parser.add_argument('file_name', type=str, help='export file name')

#only support ‘Schrodinger' for now
parser.add_argument('--scenario', type=str, help='scenario for PINNs', default='Schrodinger')


def export_sch(conf_sch, export_format, export_name):
    """
    export PINNs for Schrodinger model

    Args:
        conf_sch (dict): dictionary for configuration, see src/config.py for details
        export_format (str): file format to export
        export_name (str): name of exported file
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    num_neuron = conf_sch['num_neuron']
    layers = [2, num_neuron, num_neuron, num_neuron, num_neuron, 2]
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    n = PINNs(layers, lb, ub)
    param_dict = load_checkpoint(ck_file)
    load_param_into_net(n, param_dict)

    batch_size = conf_sch['N0'] + 2*conf_sch['Nb'] +conf_sch['Nf']
    inputs = Tensor(np.ones((batch_size, 2)), mstype.float32)
    export(n, inputs, file_name=export_name, file_format=export_format)


if __name__ == '__main__':
    args_opt = parser.parse_args()
    ck_file = args_opt.ck_file
    file_format = 'MINDIR'
    file_name = args_opt.file_name
    pinns_scenario = args_opt.scenario
    conf = config.config_Sch

    if pinns_scenario == 'Schrodinger':
        export_sch(conf, file_format, file_name)
    else:
        print(f'{pinns_scenario} scenario in PINNs is not supported to export for now')
