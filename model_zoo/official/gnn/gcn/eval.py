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

"""
GCN eval script.
"""

import argparse
import numpy as np
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint
from mindspore import Tensor
from mindspore import Model, context

from src.config import ConfigGCN
from src.dataset import get_adj_features_labels, get_mask
from src.metrics import Loss
from src.gcn import GCN

def run_gcn_infer():
    """
    Run gcn infer
    """
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--data_dir', type=str, default='./data/cora/cora_mr', help='Dataset directory')
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    parser.add_argument("--model_ckpt", type=str, required=True,
                        help="existed checkpoint address.")
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target, save_graphs=False)
    config = ConfigGCN()
    adj, feature, label_onehot, _ = get_adj_features_labels(args_opt.data_dir)
    feature_d = np.expand_dims(feature, axis=0)
    label_onehot_d = np.expand_dims(label_onehot, axis=0)
    data = {"feature": feature_d, "label": label_onehot_d}
    dataset = ds.NumpySlicesDataset(data=data)
    adj = Tensor(adj, dtype=mstype.float32)
    feature = Tensor(feature)
    nodes_num = label_onehot.shape[0]
    test_mask = get_mask(nodes_num, nodes_num - args_opt.test_nodes_num, nodes_num)
    class_num = label_onehot.shape[1]
    input_dim = feature.shape[1]
    gcn_net_test = GCN(config, input_dim, class_num, adj)
    load_checkpoint(args_opt.model_ckpt, net=gcn_net_test)
    eval_metrics = {'Acc': nn.Accuracy()}
    criterion = Loss(test_mask, config.weight_decay, gcn_net_test.trainable_params()[0])
    model = Model(gcn_net_test, loss_fn=criterion, metrics=eval_metrics)
    res = model.eval(dataset, dataset_sink_mode=True)
    print(res)

if __name__ == '__main__':
    run_gcn_infer()
