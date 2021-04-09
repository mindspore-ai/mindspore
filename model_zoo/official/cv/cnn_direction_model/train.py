# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""train CNN direction model."""
import argparse
import os
import random
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore import dataset as de
from mindspore.communication.management import init
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore.nn.optim.adam import Adam
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.cnn_direction_model import CNNDirectionModel
from src.config import config1 as config
from src.dataset import create_dataset_train

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--is_save_on_master', type=int, default=1, help='save ckpt on master or all rank')

args_opt = parser.parse_args()

random.seed(11)
np.random.seed(11)
de.config.set_seed(11)
ms.common.set_seed(11)

if __name__ == '__main__':

    target = args_opt.device_target
    ckpt_save_dir = config.save_checkpoint_path

    # init context
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_id = int(os.getenv('RANK_ID', '0'))
    rank_size = int(os.getenv('RANK_SIZE', '1'))
    run_distribute = rank_size > 1
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id, save_graphs=False)

    print("train args: ", args_opt, "\ncfg: ", config,
          "\nparallel args: rank_id {}, device_id {}, rank_size {}".format(rank_id, device_id, rank_size))

    if run_distribute:
        context.set_auto_parallel_context(device_num=rank_size, parallel_mode=ParallelMode.DATA_PARALLEL)
        init()

    args_opt.rank_save_ckpt_flag = 0
    if args_opt.is_save_on_master:
        if rank_id == 0:
            args_opt.rank_save_ckpt_flag = 1
    else:
        args_opt.rank_save_ckpt_flag = 1

    # create dataset
    dataset_name = config.dataset_name
    dataset = create_dataset_train(args_opt.dataset_path +  "/" + dataset_name +
                                   ".mindrecord0", config=config, dataset_name=dataset_name)
    step_size = dataset.get_dataset_size()

    # define net
    net = CNNDirectionModel([3, 64, 48, 48, 64], [64, 48, 48, 64, 64], [256, 64], [64, 512])

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)

    lr = config.lr
    lr = Tensor(lr, ms.float32)

    # define opt
    opt = Adam(params=net.trainable_params(), learning_rate=lr, eps=1e-07)

    # define loss, model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="sum")

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={"Accuracy": Accuracy()})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        if args_opt.rank_save_ckpt_flag == 1:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="cnn_direction_model", directory=ckpt_save_dir, config=config_ck)
            cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=False)
