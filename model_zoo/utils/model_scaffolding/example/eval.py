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
"""Evaluation process"""

import os

from mindspore import nn
from mindspore import context
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.serialization import load_checkpoint

from src.moxing_adapter import moxing_wrapper
from src.config import config
from src.dataset import create_lenet_dataset
from src.foo import LeNet5


@moxing_wrapper()
def eval_lenet5():
    """Evaluation of lenet5"""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    network = LeNet5(config.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    load_checkpoint(config.ckpt_path, network)
    ds_eval = create_lenet_dataset(os.path.join(config.data_path, "test"), config.batch_size, 1)
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))


if __name__ == '__main__':
    eval_lenet5()
