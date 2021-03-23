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
"""crnn training"""
import os
import argparse
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.wrap import WithLossCell
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_group_size, get_rank

from src.loss import CTCLoss
from src.dataset import create_dataset
from src.crnn import crnn
from src.crnn_for_train import TrainOneStepCellWithGradClip

set_seed(1)

parser = argparse.ArgumentParser(description="crnn training")
parser.add_argument("--run_distribute", action='store_true', help="Run distribute, default is false.")
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path, default is None')
parser.add_argument('--platform', type=str, default='Ascend', choices=['Ascend'],
                    help='Running platform, only support Ascend now. Default is Ascend.')
parser.add_argument('--model', type=str, default='lowercase', help="Model type, default is lowercase")
parser.add_argument('--dataset', type=str, default='synth', choices=['synth', 'ic03', 'ic13', 'svt', 'iiit5k'])
parser.set_defaults(run_distribute=False)
args_opt = parser.parse_args()

if args_opt.model == 'lowercase':
    from src.config import config1 as config
else:
    from src.config import config2 as config
context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.platform, save_graphs=False)
if args_opt.platform == 'Ascend':
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)


if __name__ == '__main__':
    lr_scale = 1
    if args_opt.run_distribute:
        if args_opt.platform == 'Ascend':
            init()
            lr_scale = 1
            device_num = int(os.environ.get("RANK_SIZE"))
            rank = int(os.environ.get("RANK_ID"))
        else:
            init()
            lr_scale = 1
            device_num = get_group_size()
            rank = get_rank()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        device_num = 1
        rank = 0

    max_text_length = config.max_text_length
    # create dataset
    dataset = create_dataset(name=args_opt.dataset, dataset_path=args_opt.dataset_path, batch_size=config.batch_size,
                             num_shards=device_num, shard_id=rank, config=config)
    step_size = dataset.get_dataset_size()
    # define lr
    lr_init = config.learning_rate
    lr = nn.dynamic_lr.cosine_decay_lr(0.0, lr_init, config.epoch_size * step_size, step_size, config.epoch_size)
    loss = CTCLoss(max_sequence_length=config.num_step,
                   max_label_length=max_text_length,
                   batch_size=config.batch_size)
    net = crnn(config)
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum, nesterov=config.nesterov)

    net = WithLossCell(net, loss)
    net = TrainOneStepCellWithGradClip(net, opt).set_train()
    # define model
    model = Model(net)
    # define callbacks
    callbacks = [LossMonitor(), TimeMonitor(data_size=step_size)]
    if config.save_checkpoint and rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
        ckpt_cb = ModelCheckpoint(prefix="crnn", directory=save_ckpt_path, config=config_ck)
        callbacks.append(ckpt_cb)
    model.train(config.epoch_size, dataset, callbacks=callbacks)
