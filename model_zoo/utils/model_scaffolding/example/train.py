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

"""Training process"""

import os

from mindspore import nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy

from src.moxing_adapter import moxing_wrapper
from src.config import config
from src.dataset import create_lenet_dataset
from src.foo import LeNet5


@moxing_wrapper()
def train_lenet5():
    """
    Train lenet5
    """
    config.ckpt_path = config.output_path
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    ds_train = create_lenet_dataset(os.path.join(config.data_path, "train"), config.batch_size, num_parallel_workers=1)
    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    network = LeNet5(config.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
                                 directory=None if config.ckpt_path == "" else config.ckpt_path,
                                 config=config_ck)

    if config.device_target != "Ascend":
        model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    else:
        model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O2")

    print("============== Starting Training ==============")
    model.train(config.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()])


if __name__ == '__main__':
    train_lenet5()
