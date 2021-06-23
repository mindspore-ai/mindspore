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
"""YoloV5 train."""
import os
import time
import argparse
import datetime
import mindspore as ms
from mindspore.context import ParallelMode
from mindspore.nn.optim.momentum import Momentum
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import _InternalCallbackParam, CheckpointConfig

from src.yolo import YOLOV5s, YoloWithLossCell, TrainingWrapper
from src.logger import get_logger
from src.util import AverageMeter, get_param_groups
from src.lr_scheduler import get_lr
from src.yolo_dataset import create_yolo_dataset
from src.initializer import default_recurisive_init, load_yolov5_params
from src.config import ConfigYOLOV5
ms.set_seed(1)


def parse_args(cloud_args=None):
    """Parse train arguments."""
    parser = argparse.ArgumentParser('mindspore coco training')

    # device related
    parser.add_argument('--device_target', type=str, default='Ascend',
                        help='device where the code will be implemented.')

    # dataset related
    parser.add_argument('--data_dir', type=str, help='Train dataset directory.')
    parser.add_argument('--per_batch_size', default=8, type=int, help='Batch size for Training. Default: 8')

    # network related
    parser.add_argument('--pretrained_backbone', default='', type=str,
                        help='The backbone file of YOLOv5. Default: "".')
    parser.add_argument('--resume_yolov5', default='', type=str,
                        help='The ckpt file of YOLOv5, which used to fine tune. Default: ""')

    # optimizer and lr related
    parser.add_argument('--lr_scheduler', default='cosine_annealing', type=str,
                        help='Learning rate scheduler, options: exponential, cosine_annealing. Default: exponential')
    parser.add_argument('--lr', default=0.013, type=float, help='Learning rate. Default: 0.01')
    parser.add_argument('--lr_epochs', type=str, default='220,250',
                        help='Epoch of changing of lr changing, split with ",". Default: 220,250')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Decrease lr by a factor of exponential lr_scheduler. Default: 0.1')
    parser.add_argument('--eta_min', type=float, default=0., help='Eta_min in cosine_annealing scheduler. Default: 0')
    parser.add_argument('--T_max', type=int, default=300, help='T-max in cosine_annealing scheduler. Default: 320')
    parser.add_argument('--max_epoch', type=int, default=300, help='Max epoch num to train the model. Default: 320')
    parser.add_argument('--warmup_epochs', default=20, type=float, help='Warmup epochs. Default: 0')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay factor. Default: 0.0005')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum. Default: 0.9')

    # loss related
    parser.add_argument('--loss_scale', type=int, default=1024, help='Static loss scale. Default: 1024')
    parser.add_argument('--label_smooth', type=int, default=0, help='Whether to use label smooth in CE. Default:0')
    parser.add_argument('--label_smooth_factor', type=float, default=0.1,
                        help='Smooth strength of original one-hot. Default: 0.1')

    # logging related
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval steps. Default: 100')
    parser.add_argument('--ckpt_path', type=str, default='outputs/', help='Checkpoint save location. Default: outputs/')
    parser.add_argument('--ckpt_interval', type=int, default=10, help='Save checkpoint interval. Default: 10')

    parser.add_argument('--is_save_on_master', type=int, default=1,
                        help='Save ckpt on master or all rank, 1 for master, 0 for all ranks. Default: 1')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=1,
                        help='Distribute train or not, 1 for yes, 0 for no. Default: 1')
    parser.add_argument('--rank', type=int, default=0, help='Local rank of distributed. Default: 0')
    parser.add_argument('--group_size', type=int, default=1, help='World size of device. Default: 1')

    # roma obs
    parser.add_argument('--train_url', type=str, default="", help='train url')
    # profiler init
    parser.add_argument('--need_profiler', type=int, default=0,
                        help='Whether use profiler. 0 for no, 1 for yes. Default: 0')

    # reset default config
    parser.add_argument('--training_shape', type=str, default="", help='Fix training shape. Default: ""')
    parser.add_argument('--resize_rate', type=int, default=10,
                        help='Resize rate for multi-scale training. Default: None')

    args, _ = parser.parse_known_args()
    args = merge_args(args, cloud_args)
    if args.lr_scheduler == 'cosine_annealing' and args.max_epoch > args.T_max:
        args.T_max = args.max_epoch

    args.lr_epochs = list(map(int, args.lr_epochs.split(',')))
    args.data_root = os.path.join(args.data_dir, 'train2017')
    args.annFile = os.path.join(args.data_dir, 'annotations/instances_train2017.json')

    devid = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target=args.device_target, save_graphs=False, device_id=devid)
    # init distributed
    if args.is_distributed:
        if args.device_target == "Ascend":
            init()
        else:
            init("nccl")
        args.rank = get_rank()
        args.group_size = get_group_size()

    # select for master rank save ckpt or all rank save, compatible for model parallel
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

    return args

def merge_args(args, cloud_args):
    args_dict = vars(args)
    if isinstance(cloud_args, dict):
        for key in cloud_args.keys():
            val = cloud_args[key]
            if key in args_dict and val:
                arg_type = type(args_dict[key])
                if arg_type is not type(None):
                    val = arg_type(val)
                args_dict[key] = val
    return args


def convert_training_shape(args_training_shape):
    training_shape = [int(args_training_shape), int(args_training_shape)]
    return training_shape


def train(cloud_args=None):
    args = parse_args(cloud_args)
    loss_meter = AverageMeter('loss')

    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)

    network = YOLOV5s(is_training=True)
    # default is kaiming-normal
    default_recurisive_init(network)
    load_yolov5_params(args, network)

    network = YoloWithLossCell(network)
    config = ConfigYOLOV5()

    config.label_smooth = args.label_smooth
    config.label_smooth_factor = args.label_smooth_factor

    if args.training_shape:
        config.multi_scale = [convert_training_shape(args.training_shape)]
    if args.resize_rate:
        config.resize_rate = args.resize_rate

    ds, data_size = create_yolo_dataset(image_dir=args.data_root, anno_path=args.annFile, is_training=True,
                                        batch_size=args.per_batch_size, max_epoch=args.max_epoch,
                                        device_num=args.group_size, rank=args.rank, config=config)
    args.logger.info('Finish loading dataset')

    args.steps_per_epoch = int(data_size / args.per_batch_size / args.group_size)

    if not args.ckpt_interval:
        args.ckpt_interval = args.steps_per_epoch

    lr = get_lr(args)

    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr),
                   momentum=args.momentum,
                   weight_decay=args.weight_decay,
                   loss_scale=args.loss_scale)

    network = TrainingWrapper(network, opt, args.loss_scale // 2)
    network.set_train()

    if args.rank_save_ckpt_flag:
        # checkpoint save
        ckpt_max_num = args.max_epoch * args.steps_per_epoch // args.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
        save_ckpt_path = os.path.join(args.outputs_dir, 'ckpt_' + str(args.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(args.rank))
        cb_params = _InternalCallbackParam()
        cb_params.train_network = network
        cb_params.epoch_num = ckpt_max_num
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)

    old_progress = -1
    t_end = time.time()
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)

    for i, data in enumerate(data_loader):
        images = data["image"]
        input_shape = images.shape[2:4]
        images = Tensor.from_numpy(images)
        batch_y_true_0 = Tensor.from_numpy(data['bbox1'])
        batch_y_true_1 = Tensor.from_numpy(data['bbox2'])
        batch_y_true_2 = Tensor.from_numpy(data['bbox3'])
        batch_gt_box0 = Tensor.from_numpy(data['gt_box1'])
        batch_gt_box1 = Tensor.from_numpy(data['gt_box2'])
        batch_gt_box2 = Tensor.from_numpy(data['gt_box3'])
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
                    'epoch[{}], iter[{}], {}, fps:{:.2f} imgs/sec, lr:{}'.format(epoch, i, loss_meter, fps, lr[i]))
            t_end = time.time()
            loss_meter.reset()
            old_progress = i

        if (i + 1) % args.steps_per_epoch == 0 and args.rank_save_ckpt_flag:
            cb_params.cur_epoch_num += 1

    args.logger.info('==========end training===============')

if __name__ == "__main__":
    train()
