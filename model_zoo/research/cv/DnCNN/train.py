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

"""train DnCNN"""

import os
import ast
import argparse
import mindspore.dataset as ds
from mindspore import nn
from mindspore import context, Model
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor

from src.model import DnCNN
from src.config import config
from src.lr_generator import step_lr
from src.data_generator import DenoisingDataset


set_seed(1)
parser = argparse.ArgumentParser(description='Mindspore DnCNN in Ascend')
parser.add_argument('--train_data', default='./data/Train400', type=str, help='the path of train dataset')
parser.add_argument('--run_modelart', default=False, type=ast.literal_eval, help='run on modelArt, default is false')
parser.add_argument('--is_distributed', default=False, type=ast.literal_eval, help='distribute training')
parser.add_argument('--device_target', type=str, default='Ascend', help='run in Ascend')
parser.add_argument('--device_id', default=0, type=int, help='device id')
# used for adapting to cloud
parser.add_argument('--data_url', default=None, help='Location of data.')
parser.add_argument('--train_url', default=None, help='Location of training outputs.')
args = parser.parse_args()

model = config.model
basic_lr = config.basic_lr
lr_gamma = config.lr_gamma
batch_size = config.batch_size
epochs = config.epoch
sigma = config.sigma
run_modelart = args.run_modelart

save_dir = os.path.join('models', model+'_' + 'sigma'+str(sigma))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if __name__ == '__main__':
    if args.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)
        if run_modelart:
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            local_input_url = os.path.join('/cache/data' + str(device_id))
            local_output_url = os.path.join('/cache/ckpt' + str(device_id))
            context.set_context(device_id=device_id)
            if device_num > 1:
                init()
                context.set_auto_parallel_context(device_num=device_num, global_rank=device_id,
                                                  parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
                args.rank = get_rank()
            else:
                args.rank = 0
            import moxing as mox
            mox.file.copy_parallel(src_url=args.data_url, dst_url=local_input_url)
            args.train_data = local_input_url
            save_dir = local_output_url
        elif args.is_distributed:
            if os.getenv('DEVICE_ID', "not_set").isdigit():
                context.set_context(device_id=int(os.getenv('DEVICE_ID')))
            init()
            args.rank = get_rank()
            args.group_size = get_group_size()
            device_num = args.group_size
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True, all_reduce_fusion_config=[2, 18])
        else:
            args.rank = 0 # record of different training process
            args.group_size = 1
            context.set_context(device_id=args.device_id)
        print('======> Building model <======')
        # define network
        dncnn = DnCNN()
        # dataset
        DDataset = DenoisingDataset(args.train_data, sigma)
        train_dataset = ds.GeneratorDataset(DDataset, ["noised_img", "noise"], shuffle=True)
        train_dataset = train_dataset.batch(config.batch_size, drop_remainder=True)
        train_data_size = train_dataset.get_dataset_size()  # num of total patches div batch_size. means num of steps in one epoch
        # loss
        criterion = nn.MSELoss(reduction='sum')
        # learning rate
        lr = step_lr(basic_lr, lr_gamma, epochs*train_data_size, train_data_size)
        # optimizer
        optimizer = nn.Adam(dncnn.trainable_params(), learning_rate=lr)
        # define model
        dncnn_model = Model(dncnn, loss_fn=criterion, optimizer=optimizer, amp_level="O3")
        # call back
        loss_cb = LossMonitor(per_print_times=train_data_size)
        time_cb = TimeMonitor(data_size=train_data_size)
        cb = [loss_cb, time_cb]
        if config.save_checkpoint:
            ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size, keep_checkpoint_max=5)
            ckpt_save_path = os.path.join(save_dir, 'ckpt' + str(args.rank) + '/')
            ckpt_cb = ModelCheckpoint(prefix="dncnn", directory=ckpt_save_path, config=ckpt_config)
            cb.append(ckpt_cb)
        print("======> start training <======")
        dncnn_model.train(epoch=epochs, train_dataset=train_dataset,
                          callbacks=cb, dataset_sink_mode=True)
        if run_modelart:
            import moxing as mox
            mox.file.copy_parallel(src_url=save_dir, dst_url=args.train_url)
        print("======> end training <======")
    else:
        raise ValueError("Unsupported device. The device should be Ascend.")
