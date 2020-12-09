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
"""Face detection train."""
import os
import time
import datetime
import argparse
import numpy as np

from mindspore import context
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore import Tensor
from mindspore.nn import Momentum
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import _InternalCallbackParam, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype


import mindspore.dataset as de



from src.FaceDetection.yolov3 import HwYolov3 as backbone_HwYolov3
from src.FaceDetection.yolo_loss import YoloLoss
from src.network_define import BuildTrainNetworkV2, TrainOneStepWithLossScaleCell
from src.lrsche_factory import warmup_step_new
from src.logging import get_logger
from src.data_preprocess import compose_map_func
from src.config import config

devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=devid)


def parse_args():
    '''parse_args'''
    parser = argparse.ArgumentParser('Yolov3 Face Detection')

    parser.add_argument('--mindrecord_path', type=str, default='', help='dataset path, e.g. /home/data.mindrecord')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')
    parser.add_argument('--local_rank', type=int, default=0, help='current rank to support distributed')
    parser.add_argument('--world_size', type=int, default=8, help='current process number to support distributed')

    args, _ = parser.parse_known_args()

    return args


def train(args):
    '''train'''
    print('=============yolov3 start trainging==================')


    # init distributed
    if args.world_size != 1:
        init()
        args.local_rank = get_rank()
        args.world_size = get_group_size()

    args.batch_size = config.batch_size
    args.warmup_lr = config.warmup_lr
    args.lr_rates = config.lr_rates
    args.lr_steps = config.lr_steps
    args.gamma = config.gamma
    args.weight_decay = config.weight_decay
    args.momentum = config.momentum
    args.max_epoch = config.max_epoch
    args.log_interval = config.log_interval
    args.ckpt_path = config.ckpt_path
    args.ckpt_interval = config.ckpt_interval

    args.outputs_dir = os.path.join(args.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    print('args.outputs_dir', args.outputs_dir)

    args.logger = get_logger(args.outputs_dir, args.local_rank)

    if args.world_size != 8:
        args.lr_steps = [i * 8 // args.world_size for i in args.lr_steps]

    if args.world_size == 1:
        args.weight_decay = 0.

    if args.world_size != 1:
        parallel_mode = ParallelMode.DATA_PARALLEL
    else:
        parallel_mode = ParallelMode.STAND_ALONE

    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=args.world_size, gradients_mean=True)
    mindrecord_path = args.mindrecord_path

    num_classes = config.num_classes
    anchors = config.anchors
    anchors_mask = config.anchors_mask
    num_anchors_list = [len(x) for x in anchors_mask]

    momentum = args.momentum
    args.logger.info('train opt momentum:{}'.format(momentum))

    weight_decay = args.weight_decay * float(args.batch_size)
    args.logger.info('real weight_decay:{}'.format(weight_decay))
    lr_scale = args.world_size / 8
    args.logger.info('lr_scale:{}'.format(lr_scale))

    # dataloader
    args.logger.info('start create dataloader')
    epoch = args.max_epoch
    ds = de.MindDataset(mindrecord_path + "0", columns_list=["image", "annotation"], num_shards=args.world_size,
                        shard_id=args.local_rank)

    ds = ds.map(input_columns=["image", "annotation"],
                output_columns=["image", "annotation", 'coord_mask_0', 'conf_pos_mask_0', 'conf_neg_mask_0',
                                'cls_mask_0', 't_coord_0', 't_conf_0', 't_cls_0', 'gt_list_0', 'coord_mask_1',
                                'conf_pos_mask_1', 'conf_neg_mask_1', 'cls_mask_1', 't_coord_1', 't_conf_1',
                                't_cls_1', 'gt_list_1', 'coord_mask_2', 'conf_pos_mask_2', 'conf_neg_mask_2',
                                'cls_mask_2', 't_coord_2', 't_conf_2', 't_cls_2', 'gt_list_2'],
                column_order=["image", "annotation", 'coord_mask_0', 'conf_pos_mask_0', 'conf_neg_mask_0',
                              'cls_mask_0', 't_coord_0', 't_conf_0', 't_cls_0', 'gt_list_0', 'coord_mask_1',
                              'conf_pos_mask_1', 'conf_neg_mask_1', 'cls_mask_1', 't_coord_1', 't_conf_1',
                              't_cls_1', 'gt_list_1', 'coord_mask_2', 'conf_pos_mask_2', 'conf_neg_mask_2',
                              'cls_mask_2', 't_coord_2', 't_conf_2', 't_cls_2', 'gt_list_2'],
                operations=compose_map_func, num_parallel_workers=16, python_multiprocessing=True)

    ds = ds.batch(args.batch_size, drop_remainder=True, num_parallel_workers=8)

    args.steps_per_epoch = ds.get_dataset_size()
    lr = warmup_step_new(args, lr_scale=lr_scale)

    ds = ds.repeat(epoch)
    args.logger.info('args.steps_per_epoch:{}'.format(args.steps_per_epoch))
    args.logger.info('args.world_size:{}'.format(args.world_size))
    args.logger.info('args.local_rank:{}'.format(args.local_rank))
    args.logger.info('end create dataloader')
    args.logger.save_args(args)
    args.logger.important_info('start create network')
    create_network_start = time.time()

    # backbone and loss
    network = backbone_HwYolov3(num_classes, num_anchors_list, args)

    criterion0 = YoloLoss(num_classes, anchors, anchors_mask[0], 64, 0, head_idx=0.0)
    criterion1 = YoloLoss(num_classes, anchors, anchors_mask[1], 32, 0, head_idx=1.0)
    criterion2 = YoloLoss(num_classes, anchors, anchors_mask[2], 16, 0, head_idx=2.0)

    # load pretrain model
    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load model {} success'.format(args.pretrained))

    train_net = BuildTrainNetworkV2(network, criterion0, criterion1, criterion2, args)

    # optimizer
    opt = Momentum(params=train_net.trainable_params(), learning_rate=Tensor(lr), momentum=momentum,
                   weight_decay=weight_decay)

    # package training process
    train_net = TrainOneStepWithLossScaleCell(train_net, opt)
    train_net.set_broadcast_flag()

    # checkpoint
    ckpt_max_num = args.max_epoch * args.steps_per_epoch // args.ckpt_interval
    train_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval, keep_checkpoint_max=ckpt_max_num)
    ckpt_cb = ModelCheckpoint(config=train_config, directory=args.outputs_dir, prefix='{}'.format(args.local_rank))
    cb_params = _InternalCallbackParam()
    cb_params.train_network = train_net
    cb_params.epoch_num = ckpt_max_num
    cb_params.cur_epoch_num = 1
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)

    train_net.set_train()
    t_end = time.time()
    t_epoch = time.time()
    old_progress = -1
    i = 0
    scale_manager = DynamicLossScaleManager(init_loss_scale=2 ** 10, scale_factor=2, scale_window=2000)

    for data in ds.create_tuple_iterator(output_numpy=True):

        batch_images = data[0]
        batch_labels = data[1]
        coord_mask_0 = data[2]
        conf_pos_mask_0 = data[3]
        conf_neg_mask_0 = data[4]
        cls_mask_0 = data[5]
        t_coord_0 = data[6]
        t_conf_0 = data[7]
        t_cls_0 = data[8]
        gt_list_0 = data[9]
        coord_mask_1 = data[10]
        conf_pos_mask_1 = data[11]
        conf_neg_mask_1 = data[12]
        cls_mask_1 = data[13]
        t_coord_1 = data[14]
        t_conf_1 = data[15]
        t_cls_1 = data[16]
        gt_list_1 = data[17]
        coord_mask_2 = data[18]
        conf_pos_mask_2 = data[19]
        conf_neg_mask_2 = data[20]
        cls_mask_2 = data[21]
        t_coord_2 = data[22]
        t_conf_2 = data[23]
        t_cls_2 = data[24]
        gt_list_2 = data[25]

        img_tensor = Tensor(batch_images, mstype.float32)
        coord_mask_tensor_0 = Tensor(coord_mask_0.astype(np.float32))
        conf_pos_mask_tensor_0 = Tensor(conf_pos_mask_0.astype(np.float32))
        conf_neg_mask_tensor_0 = Tensor(conf_neg_mask_0.astype(np.float32))
        cls_mask_tensor_0 = Tensor(cls_mask_0.astype(np.float32))
        t_coord_tensor_0 = Tensor(t_coord_0.astype(np.float32))
        t_conf_tensor_0 = Tensor(t_conf_0.astype(np.float32))
        t_cls_tensor_0 = Tensor(t_cls_0.astype(np.float32))
        gt_list_tensor_0 = Tensor(gt_list_0.astype(np.float32))

        coord_mask_tensor_1 = Tensor(coord_mask_1.astype(np.float32))
        conf_pos_mask_tensor_1 = Tensor(conf_pos_mask_1.astype(np.float32))
        conf_neg_mask_tensor_1 = Tensor(conf_neg_mask_1.astype(np.float32))
        cls_mask_tensor_1 = Tensor(cls_mask_1.astype(np.float32))
        t_coord_tensor_1 = Tensor(t_coord_1.astype(np.float32))
        t_conf_tensor_1 = Tensor(t_conf_1.astype(np.float32))
        t_cls_tensor_1 = Tensor(t_cls_1.astype(np.float32))
        gt_list_tensor_1 = Tensor(gt_list_1.astype(np.float32))

        coord_mask_tensor_2 = Tensor(coord_mask_2.astype(np.float32))
        conf_pos_mask_tensor_2 = Tensor(conf_pos_mask_2.astype(np.float32))
        conf_neg_mask_tensor_2 = Tensor(conf_neg_mask_2.astype(np.float32))
        cls_mask_tensor_2 = Tensor(cls_mask_2.astype(np.float32))
        t_coord_tensor_2 = Tensor(t_coord_2.astype(np.float32))
        t_conf_tensor_2 = Tensor(t_conf_2.astype(np.float32))
        t_cls_tensor_2 = Tensor(t_cls_2.astype(np.float32))
        gt_list_tensor_2 = Tensor(gt_list_2.astype(np.float32))

        scaling_sens = Tensor(scale_manager.get_loss_scale(), dtype=mstype.float32)

        loss0, overflow, _ = train_net(img_tensor, coord_mask_tensor_0, conf_pos_mask_tensor_0,
                                       conf_neg_mask_tensor_0, cls_mask_tensor_0, t_coord_tensor_0,
                                       t_conf_tensor_0, t_cls_tensor_0, gt_list_tensor_0,
                                       coord_mask_tensor_1, conf_pos_mask_tensor_1, conf_neg_mask_tensor_1,
                                       cls_mask_tensor_1, t_coord_tensor_1, t_conf_tensor_1,
                                       t_cls_tensor_1, gt_list_tensor_1, coord_mask_tensor_2,
                                       conf_pos_mask_tensor_2, conf_neg_mask_tensor_2,
                                       cls_mask_tensor_2, t_coord_tensor_2, t_conf_tensor_2,
                                       t_cls_tensor_2, gt_list_tensor_2, scaling_sens)

        overflow = np.all(overflow.asnumpy())
        if overflow:
            scale_manager.update_loss_scale(overflow)
        else:
            scale_manager.update_loss_scale(False)
        args.logger.info('rank[{}], iter[{}], loss[{}], overflow:{}, loss_scale:{}, lr:{}, batch_images:{}, '
                         'batch_labels:{}'.format(args.local_rank, i, loss0, overflow, scaling_sens, lr[i],
                                                  batch_images.shape, batch_labels.shape))

        # save ckpt
        cb_params.cur_step_num = i + 1  # current step number
        cb_params.batch_num = i + 2
        if args.local_rank == 0:
            ckpt_cb.step_end(run_context)

        # save Log
        if i == 0:
            time_for_graph_compile = time.time() - create_network_start
            args.logger.important_info('Yolov3, graph compile time={:.2f}s'.format(time_for_graph_compile))

        if i % args.steps_per_epoch == 0:
            cb_params.cur_epoch_num += 1

        if i % args.log_interval == 0 and args.local_rank == 0:
            time_used = time.time() - t_end
            epoch = int(i / args.steps_per_epoch)
            fps = args.batch_size * (i - old_progress) * args.world_size / time_used
            args.logger.info('epoch[{}], iter[{}], loss:[{}], {:.2f} imgs/sec'.format(epoch, i, loss0, fps))
            t_end = time.time()
            old_progress = i

        if i % args.steps_per_epoch == 0 and args.local_rank == 0:
            epoch_time_used = time.time() - t_epoch
            epoch = int(i / args.steps_per_epoch)
            fps = args.batch_size * args.world_size * args.steps_per_epoch / epoch_time_used
            args.logger.info('=================================================')
            args.logger.info('epoch time: epoch[{}], iter[{}], {:.2f} imgs/sec'.format(epoch, i, fps))
            args.logger.info('=================================================')
            t_epoch = time.time()

        i = i + 1

    args.logger.info('=============yolov3 training finished==================')


if __name__ == "__main__":
    arg = parse_args()
    train(arg)
