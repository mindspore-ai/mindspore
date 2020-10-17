# Copyright 2020 Huawei Technologies Co., Ltd
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
"""export checkpoint file into air models"""
import argparse
import numpy as np

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, export

from src.gcn import GCN
from src.config import ConfigGCN

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN_export')
    parser.add_argument('--ckpt_file', type=str, default='', help='GCN ckpt file.')
    parser.add_argument('--output_file', type=str, default='gcn.air', help='GCN output air name.')
    parser.add_argument('--dataset', type=str, default='cora', help='GCN dataset name.')
    args_opt = parser.parse_args()

    config = ConfigGCN()

    if args_opt.dataset == "cora":
        input_dim = 1433
        class_num = 7
        adj = Tensor(np.zeros((2708, 2708), np.float64))
        feature = Tensor(np.zeros((2708, 1433), np.float32))
    else:
        input_dim = 3703
        class_num = 6
        adj = Tensor(np.zeros((3312, 3312), np.float64))
        feature = Tensor(np.zeros((3312, 3703), np.float32))

    gcn_net = GCN(config, input_dim, class_num)

    gcn_net.set_train(False)
    load_checkpoint(args_opt.ckpt_file, net=gcn_net)
    gcn_net.add_flags_recursive(fp16=True)

    export(gcn_net, adj, feature, file_name=args_opt.output_file, file_format="AIR")
