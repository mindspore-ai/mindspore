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
######################## eval lenet example ########################
eval lenet according to model file:
python eval.py --data_path /YourDataPath --ckpt_path Your.ckpt
"""

import os
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.dataset import create_dataset
from src.lenet import LeNet5

import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy


def modelarts_process():
    config.ckpt_path = config.ckpt_file

@moxing_wrapper(pre_process=modelarts_process)
def eval_lenet():
    print('eval with config: ', config)
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    network = LeNet5(config.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # repeat_size = config.epoch_size
    net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(config.ckpt_path)
    load_param_into_net(network, param_dict)
    ds_eval = create_dataset(os.path.join(config.data_path, "test"),
                             config.batch_size,
                             1)
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))


if __name__ == "__main__":
    eval_lenet()
