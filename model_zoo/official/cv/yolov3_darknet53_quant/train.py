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
"""YoloV3-Darknet53-Quant train."""

import os
import time
import argparse
import datetime

from mindspore.context import ParallelMode
from mindspore.nn.optim.momentum import Momentum
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, RunContext
from mindspore.train.callback import _InternalCallbackParam, CheckpointConfig
import mindspore as ms
from mindspore.compression.quant import QuantizationAwareTraining
from mindspore.common import set_seed

from src.yolo import YOLOV3DarkNet53, YoloWithLossCell, TrainingWrapper
from src.logger import get_logger
from src.util import AverageMeter, get_param_groups
from src.lr_scheduler import get_lr
from src.yolo_dataset import create_yolo_dataset
from src.initializer import default_recurisive_init, load_yolov3_quant_params
from src.config import ConfigYOLOV3DarkNet53
from src.transforms import batch_preprocess_true_box, batch_preprocess_true_box_single
from src.util import ShapeRecord

set_seed(1)

devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                    device_target="Ascend", save_graphs=True, device_id=devid)


def parse_args():
    """Parse train arguments."""
    parser = argparse.ArgumentParser('mindspore coco training')

    # dataset related
    parser.add_argument('--data_dir', type=str, default='', help='Train data dir. Default: ""')
    parser.add_argument('--per_batch_size', default=16, type=int, help='Batch size for per device. Default: 16')

    # network related
    parser.add_argument('--resume_yolov3', default='', type=str,\
                       help='The ckpt file of yolov3-darknet53, which used to yolov3-darknet53 quant. Default: ""')

    # optimizer and lr related
    parser.add_argument('--lr_scheduler', default='exponential', type=str,\
                        help='Learning rate scheduler, option type: exponential, '
                             'cosine_annealing. Default: exponential')
    parser.add_argument('--lr', default=0.012, type=float, help='Learning rate of the training')
    parser.add_argument('--lr_epochs', type=str, default='92,105',\
                       help='Epoch of lr changing. Default: 92,105')
    parser.add_argument('--lr_gamma', type=float, default=0.1,\
                       help='Decrease lr by a factor of exponential lr_scheduler. Default: 0.1')
    parser.add_argument('--eta_min', type=float, default=0.,\
                       help='Eta_min in cosine_annealing scheduler. Default: 0.')
    parser.add_argument('--T_max', type=int, default=135,\
                       help='T-max in cosine_annealing scheduler. Default: 135')
    parser.add_argument('--max_epoch', type=int, default=135,\
                       help='Max epoch num to train the model. Default: 135')
    parser.add_argument('--warmup_epochs', type=float, default=0, help='Warmup epochs. Default: 0')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay. Default: 0.0005')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum. Default: 0.9')

    # loss related
    parser.add_argument('--loss_scale', type=int, default=1024, help='Static loss scale. Default: 1024')
    parser.add_argument('--label_smooth', type=int, default=0, help='Whether to use label smooth in CE. Default: 0')
    parser.add_argument('--label_smooth_factor', type=float, default=0.1,\
                       help='Smooth strength of original one-hot. Default: 0.1')

    # logging related
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval steps. Default: 100')
    parser.add_argument('--ckpt_path', type=str, default='outputs/',\
                       help='Checkpoint save location. Default: "outputs/"')
    parser.add_argument('--ckpt_interval', type=int, default=None, help='Save checkpoint interval. Default: None')
    parser.add_argument('--is_save_on_master', type=int, default=1,\
                       help='Save ckpt on master or all rank, 1 for master, 0 for all ranks. Default: 1')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=0,\
                       help='Distribute train or not, 1 for yes, 0 for no. Default: 0')
    parser.add_argument('--rank', type=int, default=0, help='Local rank of distributed, Default: 0')
    parser.add_argument('--group_size', type=int, default=1, help='World size of device, Default: 1')

    # profiler init
    parser.add_argument('--need_profiler', type=int, default=0,\
                       help='Whether use profiler, 1 for yes, 0 for no, Default: 0')

    # reset default config
    parser.add_argument('--training_shape', type=str, default="", help='Fix training shape. Default: ""')
    parser.add_argument('--resize_rate', type=int, default=None,\
                       help='Resize rate for multi-scale training. Default: None')

    args, _ = parser.parse_known_args()
    if args.lr_scheduler == 'cosine_annealing' and args.max_epoch > args.T_max:
        args.T_max = args.max_epoch

    args.lr_epochs = list(map(int, args.lr_epochs.split(',')))
    args.data_root = os.path.join(args.data_dir, 'train2014')
    args.annFile = os.path.join(args.data_dir, 'annotations/instances_train2014.json')

    # init distributed
    if args.is_distributed:
        init()
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
    return args


def conver_training_shape(args):
    training_shape = [int(args.training_shape), int(args.training_shape)]
    return training_shape

def build_quant_network(network):
    quantizer = QuantizationAwareTraining(bn_fold=True,
                                          per_channel=[True, False],
                                          symmetric=[True, False],
                                          one_conv_fold=False)
    network = quantizer.quantize(network)
    return network


def train():
    """Train function."""
    args = parse_args()
    args.logger.save_args(args)

    if args.need_profiler:
        from mindspore.profiler.profiling import Profiler
        profiler = Profiler(output_path=args.outputs_dir, is_detail=True, is_show_op_path=True)

    loss_meter = AverageMeter('loss')

    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)

    network = YOLOV3DarkNet53(is_training=True)
    # default is kaiming-normal
    default_recurisive_init(network)
    load_yolov3_quant_params(args, network)

    config = ConfigYOLOV3DarkNet53()
    # convert fusion network to quantization aware network
    if config.quantization_aware:
        network = build_quant_network(network)

    network = YoloWithLossCell(network)
    args.logger.info('finish get network')

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

    lr = get_lr(args)

    opt = Momentum(params=get_param_groups(network), learning_rate=Tensor(lr), momentum=args.momentum,
                   weight_decay=args.weight_decay, loss_scale=args.loss_scale)

    network = TrainingWrapper(network, opt)
    network.set_train()

    if args.rank_save_ckpt_flag:
        # checkpoint save
        ckpt_max_num = args.max_epoch * args.steps_per_epoch // args.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
        save_ckpt_path = os.path.join(args.outputs_dir, 'ckpt_' + str(args.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=save_ckpt_path, prefix='{}'.format(args.rank))
        cb_params = _InternalCallbackParam()
        cb_params.train_network = network
        cb_params.epoch_num = ckpt_max_num
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)

    old_progress = -1
    t_end = time.time()
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)

    shape_record = ShapeRecord()
    for i, data in enumerate(data_loader):
        images = data["image"]
        input_shape = images.shape[2:4]
        args.logger.info('iter[{}], shape{}'.format(i, input_shape[0]))
        shape_record.set(input_shape)

        images = Tensor.from_numpy(images)
        annos = data["annotation"]
        if args.group_size == 1:
            batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1, batch_gt_box2 = \
                batch_preprocess_true_box(annos, config, input_shape)
        else:
            batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1, batch_gt_box2 = \
                batch_preprocess_true_box_single(annos, config, input_shape)

        batch_y_true_0 = Tensor.from_numpy(batch_y_true_0)
        batch_y_true_1 = Tensor.from_numpy(batch_y_true_1)
        batch_y_true_2 = Tensor.from_numpy(batch_y_true_2)
        batch_gt_box0 = Tensor.from_numpy(batch_gt_box0)
        batch_gt_box1 = Tensor.from_numpy(batch_gt_box1)
        batch_gt_box2 = Tensor.from_numpy(batch_gt_box2)

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
