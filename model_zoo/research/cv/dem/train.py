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
"""
######################## train DEM ########################
train DEM
python train.py --data_path = /YourDataPath \
                --dataset = AwA or CUB \
                --train_mode = att, word or fusion
"""
import time
import sys
import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore import save_checkpoint
from mindspore import dataset as ds
from mindspore import Model
from mindspore import set_seed
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import init, get_rank, get_group_size

from src.dataset import dataset_AwA, dataset_CUB, SingleDataIterable, DoubleDataIterable
from src.demnet import MyTrainOneStepCell
from src.set_parser import set_parser
from src.utils import acc_cfg, backbone_cfg, param_cfg, withlosscell_cfg, save_min_acc_cfg
from src.accuracy import compute_accuracy_att, compute_accuracy_word, compute_accuracy_fusion

if __name__ == "__main__":
    # Set graph mode, device id
    set_seed(1000)
    args = set_parser()
    context.set_context(mode=context.GRAPH_MODE, \
                        device_target=args.device_target, \
                        device_id=args.device_id)
    if args.distribute:
        init()
        args.device_num = get_group_size()
        rank_id = get_rank()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.AUTO_PARALLEL, \
                                          gradients_mean=True, \
                                          device_num=args.device_num)
    else:
        rank_id = 0
    # Initialize parameters
    pred_len = acc_cfg(args)
    lr, weight_decay, clip_param = param_cfg(args)
    if np.equal(args.distribute, True):
        lr = lr * 5
    batch_size = args.batch_size

    # Loading datasets and iterators
    if args.dataset == 'AwA':
        train_x, train_att, train_word, \
        test_x, test_att, test_word, \
        test_label, test_id = dataset_AwA(args.data_path)
        if args.train_mode == 'att':
            custom_data = ds.GeneratorDataset(SingleDataIterable(train_att, train_x),
                                              ['label', 'data'],
                                              num_shards=args.device_num,
                                              shard_id=rank_id,
                                              shuffle=True)
        elif args.train_mode == 'word':
            custom_data = ds.GeneratorDataset(SingleDataIterable(train_word, train_x),
                                              ['label', 'data'],
                                              num_shards=args.device_num,
                                              shard_id=rank_id,
                                              shuffle=True)
        elif args.train_mode == 'fusion':
            custom_data = ds.GeneratorDataset(DoubleDataIterable(train_att, train_word, train_x),
                                              ['label1', 'label2', 'data'],
                                              num_shards=args.device_num,
                                              shard_id=rank_id,
                                              shuffle=True)
    elif args.dataset == 'CUB':
        train_att, train_x, \
        test_x, test_att, \
        test_label, test_id = dataset_CUB(args.data_path)
        if args.train_mode == 'att':
            custom_data = ds.GeneratorDataset(SingleDataIterable(train_att, train_x),
                                              ['label', 'data'],
                                              num_shards=args.device_num,
                                              shard_id=rank_id,
                                              shuffle=True)
        elif args.train_mode == 'word':
            print("Warning: Do not support word vector mode training in CUB dataset.")
            print("Only attribute mode is supported in this dataset.")
            sys.exit(0)
        elif args.train_mode == 'fusion':
            print("Warning: Do not support fusion mode training in CUB dataset.")
            print("Only attribute mode is supported in this dataset.")
            sys.exit(0)
    # Note: Must set "drop_remainder = True" in parallel mode.
    custom_data = custom_data.batch(batch_size, drop_remainder=True)

    # Build network
    net = backbone_cfg(args)
    loss_fn = nn.MSELoss(reduction='mean')
    optim = nn.Adam(net.trainable_params(), lr, weight_decay)
    MyWithLossCell = withlosscell_cfg(args)
    loss_net = MyWithLossCell(net, loss_fn)
    train_net = MyTrainOneStepCell(loss_net, optim)
    model = Model(train_net)

    # Train
    start = time.time()
    acc_max = 0
    save_min_acc = save_min_acc_cfg(args)
    save_ckpt = args.save_ckpt
    ckpt_file_name = save_ckpt + '/train.ckpt'
    interval_step = args.interval_step
    epoch_size = args.epoch_size
    print("============== Starting Training ==============")
    if np.equal(args.distribute, True):
        now = time.localtime()
        nowt = time.strftime("%Y-%m-%d-%H:%M:%S", now)
        print(nowt)
        loss_cb = LossMonitor(interval_step)
        ckpt_config = CheckpointConfig(save_checkpoint_steps=interval_step)
        ckpt_callback = ModelCheckpoint(prefix='auto_parallel', config=ckpt_config)
        t1 = time.time()
        model.train(epoch_size, train_dataset=custom_data, callbacks=[loss_cb, ckpt_callback], dataset_sink_mode=True)
        end = time.time()
        t3 = 1000 * (end - t1) / (88 * epoch_size)
        print('total time:', end - start)
        print('speed_8p = %.3f ms/step'%t3)
        now = time.localtime()
        nowt = time.strftime("%Y-%m-%d-%H:%M:%S", now)
        print(nowt)
    else:
        for i in range(epoch_size):
            t1 = time.time()
            model.train(1, train_dataset=custom_data, callbacks=LossMonitor(interval_step), dataset_sink_mode=True)
            t2 = time.time()
            t3 = 1000 * (t2 - t1) / 88
            if args.train_mode == 'att':
                acc = compute_accuracy_att(net, pred_len, test_att, test_x, test_id, test_label)
            elif args.train_mode == 'word':
                acc = compute_accuracy_word(net, pred_len, test_word, test_x, test_id, test_label)
            else:
                acc = compute_accuracy_fusion(net, pred_len, test_att, test_word, test_x, test_id, test_label)
            if acc > acc_max:
                acc_max = acc
                if acc_max > save_min_acc:
                    save_checkpoint(net, ckpt_file_name)
            print('epoch:', i + 1, 'accuracy = %.5f'%acc, 'speed = %.3f ms/step'%t3)
        end = time.time()
        print("total time:", end - start)
