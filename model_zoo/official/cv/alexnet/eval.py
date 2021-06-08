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
######################## eval alexnet example ########################
eval alexnet according to model file:
python eval.py --data_path /YourDataPath --ckpt_path Your.ckpt
"""

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.dataset import create_dataset_cifar10, create_dataset_imagenet
from src.alexnet import AlexNet

import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.communication.management import init


def modelarts_process():
    config.ckpt_path = config.ckpt_file

@moxing_wrapper(pre_process=modelarts_process)
def eval_alexnet():
    print("============== Starting Testing ==============")
    device_num = get_device_num()
    if device_num > 1:
        # context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
        context.set_context(mode=context.GRAPH_MODE, device_target='Davinci', save_graphs=False)
        if config.device_target == "Ascend":
            context.set_context(device_id=get_device_id())
            init()
        elif config.device_target == "GPU":
            init()

    if config.dataset_name == 'cifar10':
        network = AlexNet(config.num_classes, phase='test')
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        opt = nn.Momentum(network.trainable_params(), config.learning_rate, config.momentum)
        ds_eval = create_dataset_cifar10(config, config.data_path, config.batch_size, status="test", \
            target=config.device_target)
        param_dict = load_checkpoint(config.ckpt_path)
        print("load checkpoint from [{}].".format(config.ckpt_path))
        load_param_into_net(network, param_dict)
        network.set_train(False)
        model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})

    elif config.dataset_name == 'imagenet':
        network = AlexNet(config.num_classes, phase='test')
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        ds_eval = create_dataset_imagenet(config, config.data_path, config.batch_size, training=False)
        param_dict = load_checkpoint(config.ckpt_path)
        print("load checkpoint from [{}].".format(config.ckpt_path))
        load_param_into_net(network, param_dict)
        network.set_train(False)
        model = Model(network, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    else:
        raise ValueError("Unsupported dataset.")

    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    result = model.eval(ds_eval, dataset_sink_mode=config.dataset_sink_mode)
    print("result : {}".format(result))


if __name__ == "__main__":
    eval_alexnet()
