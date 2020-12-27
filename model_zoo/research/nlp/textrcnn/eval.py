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
"""model evaluation script"""
import os
import argparse
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import LossMonitor
from mindspore.common import set_seed

from src.config import textrcnn_cfg as cfg
from src.dataset import create_dataset
from src.textrcnn import textrcnn

set_seed(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='textrcnn')
    parser.add_argument('--ckpt_path', type=str)
    args = parser.parse_args()
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target="Ascend")

    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)

    embedding_table = np.loadtxt(os.path.join(cfg.preprocess_path, "weight.txt")).astype(np.float32)
    network = textrcnn(weight=Tensor(embedding_table), vocab_size=embedding_table.shape[0],
                       cell=cfg.cell, batch_size=cfg.batch_size)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    loss_cb = LossMonitor()
    print("============== Starting Testing ==============")
    ds_eval = create_dataset(cfg.preprocess_path, cfg.batch_size, False)
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    network.set_train(False)
    model = Model(network, loss, metrics={'acc': Accuracy()}, amp_level='O3')
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))
