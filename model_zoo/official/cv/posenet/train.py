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
"""train posenet"""
import ast
import argparse
import os
import time

from mindspore import context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.nn import Adagrad
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.config import common_config, KingsCollege, StMarysChurch
from src.dataset import data_to_mindrecord, create_posenet_dataset
from src.loss import PosenetWithLoss
from src.posenet import PoseTrainOneStepCell

set_seed(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Posenet train.')
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is false.")
    parser.add_argument('--device_id', type=int, default=None,
                        help='device id of GPU or Ascend. (Default: None)')
    parser.add_argument('--dataset', type=str, default='KingsCollege',
                        choices=['KingsCollege', 'StMarysChurch'],
                        help='Name of dataset.')
    parser.add_argument('--device_num', type=int, default=1, help='Number of device.')
    parser.add_argument('--is_modelarts', type=ast.literal_eval, default=False, help='Train in Modelarts.')
    parser.add_argument('--data_url', default=None, help='Location of data.')
    parser.add_argument('--train_url', default=None, help='Location of training outputs.')
    parser.add_argument('--device_target', type=str, default='Ascend',
                        choices=['Ascend', 'GPU'],
                        help='Name of device target.')
    args_opt = parser.parse_args()

    cfg = common_config
    if args_opt.dataset == "KingsCollege":
        dataset_cfg = KingsCollege
    elif args_opt.dataset == "StMarysChurch":
        dataset_cfg = StMarysChurch

    device_target = args_opt.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    if args_opt.run_distribute:
        if device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
            init()
        elif device_target == "GPU":
            init()
            context.set_auto_parallel_context(device_num=args_opt.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
    else:
        if args_opt.device_id is not None:
            context.set_context(device_id=args_opt.device_id)
        else:
            context.set_context(device_id=cfg.device_id)

    train_dataset_path = dataset_cfg.dataset_path
    if args_opt.is_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=args_opt.data_url,
                               dst_url='/cache/dataset_train/device_' + os.getenv('DEVICE_ID'))
        train_dataset_path = '/cache/dataset_train/device_' + os.getenv('DEVICE_ID') + '/'

    # It will generate train mindrecord file in cfg.mindrecord_dir,
    # and the file name is "dataset_cfg.name + _posenet_train.mindrecord".
    prefix = "_posenet_train.mindrecord"
    mindrecord_dir = dataset_cfg.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, dataset_cfg.name + prefix)
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create mindrecord for train.")
        data_to_mindrecord(train_dataset_path, True, mindrecord_file)
        print("Create mindrecord done, at {}".format(mindrecord_dir))
    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    dataset = create_posenet_dataset(mindrecord_file, batch_size=dataset_cfg.batch_size,
                                     device_num=args_opt.device_num, is_training=True)
    step_per_epoch = dataset.get_dataset_size()

    net_with_loss = PosenetWithLoss(cfg.pre_trained)
    opt = Adagrad(params=net_with_loss.trainable_params(),
                  learning_rate=dataset_cfg.lr_init,
                  weight_decay=dataset_cfg.weight_decay)
    net_with_grad = PoseTrainOneStepCell(net_with_loss, opt)
    model = Model(net_with_grad)

    time_cb = TimeMonitor(data_size=step_per_epoch)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if cfg.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_epochs * step_per_epoch,
                                     keep_checkpoint_max=cfg.keep_checkpoint_max)
        if args_opt.is_modelarts:
            save_checkpoint_path = '/cache/train_output/checkpoint'
            if args_opt.device_num == 1:
                ckpt_cb = ModelCheckpoint(prefix='train_posenet_' + args_opt.dataset,
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
            if args_opt.device_num > 1 and get_rank() % 8 == 0:
                ckpt_cb = ModelCheckpoint(prefix='train_posenet_' + args_opt.dataset,
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
        else:
            save_checkpoint_path = cfg.checkpoint_dir
            if not os.path.isdir(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)

            if args_opt.device_num == 1:
                ckpt_cb = ModelCheckpoint(prefix='train_posenet_' + args_opt.dataset,
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
            if args_opt.device_num > 1 and get_rank() % 8 == 0:
                ckpt_cb = ModelCheckpoint(prefix='train_posenet_' + args_opt.dataset,
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]

    epoch_size = cfg.max_steps // args_opt.device_num // step_per_epoch
    model.train(epoch_size, dataset, callbacks=cb)
    print("Train success!")

    if args_opt.is_modelarts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args_opt.train_url)
