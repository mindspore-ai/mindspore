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
"""YoloV3 train."""
import os
import time
import argparse
import datetime

from mindspore import ParallelMode
from mindspore.nn.optim.momentum import Momentum
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import _InternalCallbackParam, CheckpointConfig
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.yolo import YOLOV3DarkNet53, YoloWithLossCell, TrainingWrapper
from src.logger import get_logger
from src.util import AverageMeter, load_backbone, get_param_groups
from src.lr_scheduler import warmup_step_lr, warmup_cosine_annealing_lr, \
    warmup_cosine_annealing_lr_V2, warmup_cosine_annealing_lr_sample
from src.yolo_dataset import create_yolo_dataset
from src.initializer import default_recurisive_init
from src.config import ConfigYOLOV3DarkNet53
from src.transforms import batch_preprocess_true_box, batch_preprocess_true_box_single
from src.util import ShapeRecord


devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                    device_target="Ascend", save_graphs=True, device_id=devid)


class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss


def parse_args():
    """Parse train arguments."""
    parser = argparse.ArgumentParser('mindspore coco training')

    # dataset related
    parser.add_argument('--data_dir', type=str, default='', help='train data dir')
    parser.add_argument('--per_batch_size', default=32, type=int, help='batch size for per gpu')

    # network related
    parser.add_argument('--pretrained_backbone', default='', type=str, help='model_path, local pretrained backbone'
                                                                            ' model to load')
    parser.add_argument('--resume_yolov3', default='', type=str, help='path of pretrained yolov3')

    # optimizer and lr related
    parser.add_argument('--lr_scheduler', default='exponential', type=str,
                        help='lr-scheduler, option type: exponential, cosine_annealing')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate of the training')
    parser.add_argument('--lr_epochs', type=str, default='220,250', help='epoch of lr changing')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='decrease lr by a factor of exponential lr_scheduler')
    parser.add_argument('--eta_min', type=float, default=0., help='eta_min in cosine_annealing scheduler')
    parser.add_argument('--T_max', type=int, default=320, help='T-max in cosine_annealing scheduler')
    parser.add_argument('--max_epoch', type=int, default=320, help='max epoch num to train the model')
    parser.add_argument('--warmup_epochs', default=0, type=float, help='warmup epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # loss related
    parser.add_argument('--loss_scale', type=int, default=1024, help='static loss scale')
    parser.add_argument('--label_smooth', type=int, default=0, help='whether to use label smooth in CE')
    parser.add_argument('--label_smooth_factor', type=float, default=0.1, help='smooth strength of original one-hot')

    # logging related
    parser.add_argument('--log_interval', type=int, default=100, help='logging interval')
    parser.add_argument('--ckpt_path', type=str, default='outputs/', help='checkpoint save location')
    parser.add_argument('--ckpt_interval', type=int, default=None, help='ckpt_interval')

    parser.add_argument('--is_save_on_master', type=int, default=1, help='save ckpt on master or all rank')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=1, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

    # roma obs
    parser.add_argument('--train_url', type=str, default="", help='train url')

    # profiler init
    parser.add_argument('--need_profiler', type=int, default=0, help='whether use profiler')

    # reset default config
    parser.add_argument('--training_shape', type=str, default="", help='fix training shape')
    parser.add_argument('--resize_rate', type=int, default=None, help='resize rate for multi-scale training')

    args, _ = parser.parse_known_args()
    if args.lr_scheduler == 'cosine_annealing' and args.max_epoch > args.T_max:
        args.T_max = args.max_epoch

    args.lr_epochs = list(map(int, args.lr_epochs.split(',')))
    args.data_root = os.path.join(args.data_dir, 'train2014')
    args.annFile = os.path.join(args.data_dir, 'annotations/instances_train2014.json')

    return args


def conver_training_shape(args):
    training_shape = [int(args.training_shape), int(args.training_shape)]
    return training_shape


def train():
    """Train function."""
    args = parse_args()

    # init distributed
    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()

    # select for master rank save ckpt or all rank save, compatiable for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1

    # logger
    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.rank)
    args.logger.save_args(args)

    if args.need_profiler:
        from mindinsight.profiler.profiling import Profiler
        profiler = Profiler(output_path=args.outputs_dir, is_detail=True, is_show_op_path=True)

    loss_meter = AverageMeter('loss')

    context.reset_auto_parallel_context()
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    else:
        parallel_mode = ParallelMode.STAND_ALONE
        degree = 1
    context.set_auto_parallel_context(parallel_mode=parallel_mode, mirror_mean=True, device_num=degree)

    network = YOLOV3DarkNet53(is_training=True)
    # default is kaiming-normal
    default_recurisive_init(network)

    if args.pretrained_backbone:
        network = load_backbone(network, args.pretrained_backbone, args)
        args.logger.info('load pre-trained backbone {} into network'.format(args.pretrained_backbone))
    else:
        args.logger.info('Not load pre-trained backbone, please be careful')

    if args.resume_yolov3:
        param_dict = load_checkpoint(args.resume_yolov3)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
                args.logger.info('in resume {}'.format(key))
            else:
                param_dict_new[key] = values
                args.logger.info('in resume {}'.format(key))

        args.logger.info('resume finished')
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.resume_yolov3))

    network = YoloWithLossCell(network)
    args.logger.info('finish get network')

    config = ConfigYOLOV3DarkNet53()

    config.label_smooth = args.label_smooth
    config.label_smooth_factor = args.label_smooth_factor

    if args.training_shape:
        config.multi_scale = [conver_training_shape(args)]
    if args.resize_rate:
        config.resize_rate = args.resize_rate

    ds, data_size = create_yolo_dataset(image_dir=args.data_root, anno_path=args.annFile, is_training=True,
                                        batch_size=args.per_batch_size, max_epoch=args.max_epoch,
                                        device_num=args.group_size, rank=args.rank, config=config)
    args.logger.info('Finish loading dataset')

    args.steps_per_epoch = int(data_size / args.per_batch_size / args.group_size)

    if not args.ckpt_interval:
        args.ckpt_interval = args.steps_per_epoch

    # lr scheduler
    if args.lr_scheduler == 'exponential':
        lr = warmup_step_lr(args.lr,
                            args.lr_epochs,
                            args.steps_per_epoch,
                            args.warmup_epochs,
                            args.max_epoch,
                            gamma=args.lr_gamma,
                            )
    elif args.lr_scheduler == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(args.lr,
                                        args.steps_per_epoch,
                                        args.warmup_epochs,
                                        args.max_epoch,
                                        args.T_max,
                                        args.eta_min)
    elif args.lr_scheduler == 'cosine_annealing_V2':
        lr = warmup_cosine_annealing_lr_V2(args.lr,
                                           args.steps_per_epoch,
                                           args.warmup_epochs,
                                           args.max_epoch,
                                           args.T_max,
                                           args.eta_min)
    elif args.lr_scheduler == 'cosine_annealing_sample':
        lr = warmup_cosine_annealing_lr_sample(args.lr,
                                               args.steps_per_epoch,
                                               args.warmup_epochs,
                                               args.max_epoch,
                                               args.T_max,
                                               args.eta_min)
    else:
        raise NotImplementedError(args.lr_scheduler)

    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr),
                   momentum=args.momentum,
                   weight_decay=args.weight_decay,
                   loss_scale=args.loss_scale)

    network = TrainingWrapper(network, opt)
    network.set_train()

    if args.rank_save_ckpt_flag:
        # checkpoint save
        ckpt_max_num = args.max_epoch * args.steps_per_epoch // args.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=args.outputs_dir,
                                  prefix='{}'.format(args.rank))
        cb_params = _InternalCallbackParam()
        cb_params.train_network = network
        cb_params.epoch_num = ckpt_max_num
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)

    old_progress = -1
    t_end = time.time()
    data_loader = ds.create_dict_iterator()

    shape_record = ShapeRecord()
    for i, data in enumerate(data_loader):
        images = data["image"]
        input_shape = images.shape[2:4]
        args.logger.info('iter[{}], shape{}'.format(i, input_shape[0]))
        shape_record.set(input_shape)

        images = Tensor(images)
        annos = data["annotation"]
        if args.group_size == 1:
            batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1, batch_gt_box2 = \
                batch_preprocess_true_box(annos, config, input_shape)
        else:
            batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1, batch_gt_box2 = \
                batch_preprocess_true_box_single(annos, config, input_shape)

        batch_y_true_0 = Tensor(batch_y_true_0)
        batch_y_true_1 = Tensor(batch_y_true_1)
        batch_y_true_2 = Tensor(batch_y_true_2)
        batch_gt_box0 = Tensor(batch_gt_box0)
        batch_gt_box1 = Tensor(batch_gt_box1)
        batch_gt_box2 = Tensor(batch_gt_box2)

        input_shape = Tensor(tuple(input_shape[::-1]), ms.float32)
        loss = network(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,
                       batch_gt_box2, input_shape)
        loss_meter.update(loss.asnumpy())

        if args.rank_save_ckpt_flag:
            # ckpt progress
            cb_params.cur_step_num = i + 1  # current step number
            cb_params.batch_num = i + 2
            ckpt_cb.step_end(run_context)

        if i % args.log_interval == 0:
            time_used = time.time() - t_end
            epoch = int(i / args.steps_per_epoch)
            fps = args.per_batch_size * (i - old_progress) * args.group_size / time_used
            if args.rank == 0:
                args.logger.info(
                    'epoch[{}], iter[{}], {}, {:.2f} imgs/sec, lr:{}'.format(epoch, i, loss_meter, fps, lr[i]))
            t_end = time.time()
            loss_meter.reset()
            old_progress = i

        if (i + 1) % args.steps_per_epoch == 0 and args.rank_save_ckpt_flag:
            cb_params.cur_epoch_num += 1

        if args.need_profiler:
            if i == 10:
                profiler.analyse()
                break

    args.logger.info('==========end training===============')


if __name__ == "__main__":
    train()
