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

from src.gat import GAT
from src.config import GatConfig

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT_export')
    parser.add_argument('--ckpt_file', type=str, default='./ckpts/gat.ckpt', help='GAT ckpt file.')
    parser.add_argument('--output_file', type=str, default='gat.air', help='GAT output air name.')
    parser.add_argument('--dataset', type=str, default='cora', help='GAT dataset name.')
    args_opt = parser.parse_args()

    if args_opt.dataset == "citeseer":
        feature_size = [1, 3312, 3703]
        biases_size = [1, 3312, 3312]
        num_classes = 6
    else:
        feature_size = [1, 2708, 1433]
        biases_size = [1, 2708, 2708]
        num_classes = 7

    hid_units = GatConfig.hid_units
    n_heads = GatConfig.n_heads

    feature = np.random.uniform(0.0, 1.0, size=feature_size).astype(np.float32)
    biases = np.random.uniform(0.0, 1.0, size=biases_size).astype(np.float64)

    feature_size = feature.shape[2]
    num_nodes = feature.shape[1]

    gat_net = GAT(feature_size,
                  num_classes,
                  num_nodes,
                  hid_units,
                  n_heads,
                  attn_drop=0.0,
                  ftr_drop=0.0)

    gat_net.set_train(False)
    load_checkpoint(args_opt.ckpt_file, net=gat_net)
    gat_net.add_flags_recursive(fp16=True)

    export(gat_net, Tensor(feature), Tensor(biases), file_name=args_opt.output_file, file_format="AIR")
