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
import argparse
import os
import ast
import numpy as np

import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset as ds
import mindspore.nn as nn

from mindspore import Tensor, context
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from original_model import Net
set_seed(1234)
def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument("--epoch", help="Path to custom detections.", type=int, default=100)
    parser.add_argument("--batch_size", help="Batch size for Training.", type=int, default=8)
    parser.add_argument("--num_parallel_workers", help="The number of parallel workers.", type=int, default=16)
    parser.add_argument("--pre_train", help='The ckpt file of model.', type=str, default=None)
    parser.add_argument("--save_check_point", help="Whether save the training resulting.", type=bool, default=True)

    #learning rate
    parser.add_argument("--learning_rate", help="Learning rate.", type=float, default=0.1)
    parser.add_argument("--decay_epoch", help="decay epochs.", type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.10, help='learning rate decay.')
    parser.add_argument("--momentum", help="", type=float, default=0.9)

    #run on where
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend. (Default: 0)')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    parser.add_argument('--run_modelarts', type=ast.literal_eval, default=True, help='Run distribute')

    return parser.parse_args()
def get_lr(base_lr, total_epochs, steps_per_epoch, step_size, gamma):
    lr_each_step = []
    for i in range(1, total_epochs+1):
        if i % step_size == 0:
            base_lr *= gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(base_lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


args = parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

if args.run_modelarts:
    import moxing as mox
    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))
    args.batch_size = args.batch_size*int(8/device_num)
    context.set_context(device_id=device_id)
    local_data_url = '/cache/data'
    local_train_url = '/cache/train'
    mox.file.copy_parallel(args.data_url, local_data_url)
    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num,\
             parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    DATA_DIR = local_data_url + '/'
else:
    if args.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        args.batch_size = args.batch_size*int(8/device_num)
        context.set_context(device_id=device_id)
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,\
             parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    else:
        context.set_context(device_id=args.device_id)
        device_num = 1
        args.batch_size = args.batch_size*int(8/device_num)
        device_id = args.device_id
    DATA_DIR = args.data_url + '/'

data = ds.ImageFolderDataset(DATA_DIR, decode=True, shuffle=True,\
     num_parallel_workers=args.num_parallel_workers, num_shards=device_num, shard_id=device_id)

transform_img = [
    C.RandomCrop((128, 64), padding=4),
    C.RandomHorizontalFlip(prob=0.5),
    C.Normalize([0.485*255, 0.456*255, 0.406*255], [0.229*255, 0.224*255, 0.225*255]),
    C.HWC2CHW()
        ]

num_classes = max(data.num_classes(), 0)

data = data.map(input_columns="image", operations=transform_img, num_parallel_workers=args.num_parallel_workers)
data = data.batch(batch_size=args.batch_size)

data_size = data.get_dataset_size()

loss_cb = LossMonitor(data_size)
time_cb = TimeMonitor(data_size=data_size)
callbacks = [time_cb, loss_cb]

#save training results
if args.save_check_point and (device_num == 1 or device_id == 0):

    model_save_path = './ckpt_' + str(6) + '/'
    config_ck = CheckpointConfig(
        save_checkpoint_steps=data_size*args.epoch, keep_checkpoint_max=args.epoch)

    if args.run_modelarts:
        ckpoint_cb = ModelCheckpoint(prefix='deepsort', directory=local_train_url, config=config_ck)
    else:
        ckpoint_cb = ModelCheckpoint(prefix='deepsort', directory=model_save_path, config=config_ck)
    callbacks += [ckpoint_cb]

#design learning rate
lr = Tensor(get_lr(args.learning_rate, args.epoch, data_size, args.decay_epoch, args.gamma))
# net definition
net = Net(num_classes=num_classes)

# loss and optimizer

loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=args.momentum)
#optimizer = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=args.momentum, weight_decay=5e-4)
#optimizer = mindspore.nn.Momentum(params = net.trainable_params(), learning_rate=lr, momentum=args.momentum)

#train
model = Model(net, loss_fn=loss, optimizer=optimizer)

model.train(args.epoch, data, callbacks=callbacks, dataset_sink_mode=True)
if args.run_modelarts:
    mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
