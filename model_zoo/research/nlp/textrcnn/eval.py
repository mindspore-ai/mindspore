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
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.serialization import load_checkpoint, load_param_into_net
# from mindspore.train.callback import LossMonitor
from mindspore.common import set_seed

from src.dataset import create_dataset
from src.textrcnn import textrcnn
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config as cfg
from src.model_utils.device_adapter import get_device_id

set_seed(1)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    cfg.ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.ckpt_path)
    cfg.preprocess_path = cfg.data_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    '''eval function.'''
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target="Ascend")

    device_id = get_device_id()
    context.set_context(device_id=device_id)

    embedding_table = np.loadtxt(os.path.join(cfg.preprocess_path, "weight.txt")).astype(np.float32)
    network = textrcnn(weight=Tensor(embedding_table), vocab_size=embedding_table.shape[0],
                       cell=cfg.cell, batch_size=cfg.batch_size)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    eval_net = nn.WithEvalCell(network, loss, True)
    # loss_cb = LossMonitor()
    print("============== Starting Testing ==============")
    ds_eval = create_dataset(cfg.preprocess_path, cfg.batch_size, False)
    param_dict = load_checkpoint(cfg.ckpt_path)
    load_param_into_net(network, param_dict)
    network.set_train(False)
    model = Model(network, loss, metrics={'acc': Accuracy()}, eval_network=eval_net, eval_indexes=[0, 1, 2])
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))

if __name__ == '__main__':
    run_eval()
