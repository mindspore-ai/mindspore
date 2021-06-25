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

import os
import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, Model, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from src.dataset import create_dataset
from src.unet3d_model import UNet3d, UNet3d_
from src.lr_schedule import dynamic_lr
from src.loss import SoftmaxCrossEntropyWithLogits
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num

if config.device_target == 'Ascend':
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, \
                        device_id=device_id)
else:
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
mindspore.set_seed(1)

@moxing_wrapper()
def train_net(data_path,
              run_distribute):
    data_dir = data_path + "/image/"
    seg_dir = data_path + "/seg/"
    if run_distribute:
        init()
        if config.device_target == 'Ascend':
            rank_id = get_device_id()
            rank_size = get_device_num()
        else:
            rank_id = get_rank()
            rank_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=rank_size,
                                          gradients_mean=True)
    else:
        rank_id = 0
        rank_size = 1
    train_dataset = create_dataset(data_path=data_dir, seg_path=seg_dir, \
                                    rank_size=rank_size, rank_id=rank_id, is_training=True)
    train_data_size = train_dataset.get_dataset_size()
    print("train dataset length is:", train_data_size)

    if config.device_target == 'Ascend':
        network = UNet3d()
    else:
        network = UNet3d_()

    loss = SoftmaxCrossEntropyWithLogits()
    lr = Tensor(dynamic_lr(config, train_data_size), mstype.float32)
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=lr)
    scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    network.set_train()

    if config.device_target == 'GPU' and config.enable_fp16_gpu:
        model = Model(network, loss_fn=loss, optimizer=optimizer, loss_scale_manager=scale_manager, amp_level='O2')
    else:
        model = Model(network, loss_fn=loss, optimizer=optimizer, loss_scale_manager=scale_manager)

    time_cb = TimeMonitor(data_size=train_data_size)
    loss_cb = LossMonitor()
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    ckpoint_cb = ModelCheckpoint(prefix='Unet3d',
                                 directory=ckpt_save_dir+'./ckpt_{}/'.format(rank_id),
                                 config=ckpt_config)
    callbacks_list = [loss_cb, time_cb, ckpoint_cb]
    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset, callbacks=callbacks_list)
    print("============== End Training ==============")

if __name__ == '__main__':
    train_net(data_path=config.data_path,
              run_distribute=config.run_distribute)
