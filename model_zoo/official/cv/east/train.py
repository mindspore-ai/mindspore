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
import argparse
import datetime

from mindspore.context import ParallelMode
from mindspore.nn.optim.adam import Adam
from mindspore import Tensor, Model
from mindspore import context
from mindspore.communication.management import init
import mindspore as ms
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from mindspore.profiler.profiling import Profiler
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor

from src.util import AverageMeter, get_param_groups
from src.east import EAST, EastWithLossCell
from src.logger import get_logger
from src.initializer import default_recurisive_init
from src.dataset import create_east_dataset
from src.lr_scheduler import get_lr

set_seed(1)

parser = argparse.ArgumentParser('mindspore icdar training')

# device related
parser.add_argument(
    '--device_target',
    type=str,
    default='Ascend',
    help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument(
    '--device_id',
    type=int,
    default=0,
    help='device where the code will be implemented. (Default: Ascend)')

# dataset related
parser.add_argument(
    '--data_dir',
    default='/data/icdar2015/Training/',
    type=str,
    help='Train dataset directory.')
parser.add_argument(
    '--per_batch_size',
    default=24,
    type=int,
    help='Batch size for Training. Default: 24.')
parser.add_argument(
    '--outputs_dir',
    default='outputs/',
    type=str,
    help='output dir. Default: outputs/')

# network related
parser.add_argument(
    '--pretrained_backbone',
    default='/data/vgg/0-150_5004.ckpt',
    type=str,
    help='The ckpt file of ResNet. Default: "".')
parser.add_argument(
    '--resume_east',
    default='',
    type=str,
    help='The ckpt file of EAST, which used to fine tune. Default: ""')

# optimizer and lr related
parser.add_argument(
    '--lr_scheduler',
    default='my_lr',
    type=str,
    help='Learning rate scheduler, options: exponential, cosine_annealing. Default: cosine_annealing')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate. Default: 0.001')
parser.add_argument('--per_step', default=2, type=float,
                    help='Learning rate change times. Default: 2')
parser.add_argument(
    '--lr_gamma',
    type=float,
    default=0.1,
    help='Decrease lr by a factor of exponential lr_scheduler. Default: 0.1')
parser.add_argument(
    '--eta_min',
    type=float,
    default=0.,
    help='Eta_min in cosine_annealing scheduler. Default: 0.')
parser.add_argument(
    '--t_max',
    type=int,
    default=100,
    help='T-max in cosine_annealing scheduler. Default: 100')
parser.add_argument('--max_epoch', type=int, default=600,
                    help='Max epoch num to train the model. Default: 100')
parser.add_argument(
    '--warmup_epochs',
    default=6,
    type=float,
    help='Warmup epochs. Default: 6')
parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.0005,
    help='Weight decay factor. Default: 0.0005')

# loss related
parser.add_argument('--loss_scale', type=int, default=1,
                    help='Static loss scale. Default: 64')
parser.add_argument(
    '--lr_epochs',
    type=str,
    default='7,7',
    help='Epoch of changing of lr changing, split with ",". Default: 220,250')

# logging related
parser.add_argument('--log_interval', type=int, default=10,
                    help='Logging interval steps. Default: 100')
parser.add_argument(
    '--ckpt_path',
    type=str,
    default='outputs/',
    help='Checkpoint save location. Default: outputs/')
parser.add_argument(
    '--ckpt_interval',
    type=int,
    default=1000,
    help='Save checkpoint interval. Default: None')

parser.add_argument(
    '--is_save_on_master',
    type=int,
    default=1,
    help='Save ckpt on master or all rank, 1 for master, 0 for all ranks. Default: 1')

# distributed related
parser.add_argument(
    '--is_distributed',
    type=int,
    default=0,
    help='Distribute train or not, 1 for yes, 0 for no. Default: 1')
parser.add_argument(
    '--rank',
    type=int,
    default=0,
    help='Local rank of distributed. Default: 0')
parser.add_argument('--group_size', type=int, default=1,
                    help='World size of device. Default: 1')

# profiler init
parser.add_argument(
    '--need_profiler',
    type=int,
    default=0,
    help='Whether use profiler. 0 for no, 1 for yes. Default: 0')
# modelArts
parser.add_argument(
    '--is_modelArts',
    type=int,
    default=0,
    help='Trainning in modelArts or not, 1 for yes, 0 for no. Default: 0')

args, _ = parser.parse_known_args()

args.rank = args.device_id

# init distributed
if args.is_distributed:
    if args.device_target == "Ascend":
        init()
    else:
        init("nccl")
    args.rank = int(os.getenv('DEVICE_ID'))
    args.group_size = int(os.getenv('RANK_SIZE'))

context.set_context(
    mode=context.GRAPH_MODE,
    enable_auto_mixed_precision=True,
    device_target=args.device_target,
    save_graphs=False,
    device_id=args.rank)

# select for master rank save ckpt or all rank save, compatible for model
# parallel
args.rank_save_ckpt_flag = 0
if args.is_save_on_master:
    if args.rank == 0:
        args.rank_save_ckpt_flag = 1
else:
    args.rank_save_ckpt_flag = 1

if args.is_modelArts:
    import moxing as mox

    local_data_url = os.path.join('/cache/data', str(args.rank))
    local_ckpt_url = os.path.join('/cache/ckpt', str(args.rank))
    local_ckpt_url = os.path.join(local_ckpt_url, 'backbone.ckpt')

    mox.file.rename(args.pretrained_backbone, local_ckpt_url)
    args.pretrained_backbone = local_ckpt_url

    mox.file.copy_parallel(args.data_dir, local_data_url)
    args.data_dir = local_data_url

    args.outputs_dir = os.path.join('/cache', args.outputs_dir)

args.data_root = os.path.abspath(os.path.join(args.data_dir, 'image'))
args.txt_root = os.path.abspath(os.path.join(args.data_dir, 'groundTruth'))

outputs_dir = os.path.join(args.outputs_dir, str(args.rank))
args.outputs_dir = os.path.join(
    args.outputs_dir,
    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
args.logger = get_logger(args.outputs_dir, args.rank)
args.logger.save_args(args)

if __name__ == "__main__":
    if args.need_profiler:
        profiler = Profiler(
            output_path=args.outputs_dir,
            is_detail=True,
            is_show_op_path=True)

    loss_meter = AverageMeter('loss')

    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    degree = 1
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = int(os.getenv('RANK_SIZE'))
    context.set_auto_parallel_context(
        parallel_mode=parallel_mode,
        gradients_mean=True,
        device_num=degree)

    network = EAST()
    # default is kaiming-normal
    default_recurisive_init(network)

    # load pretrained_backbone
    if args.pretrained_backbone:
        parm_dict = load_checkpoint(args.pretrained_backbone)
        load_param_into_net(network, parm_dict)
        args.logger.info('finish load pretrained_backbone')

    network = EastWithLossCell(network)
    if args.resume_east:
        parm_dict = load_checkpoint(args.resume_east)
        load_param_into_net(network, parm_dict)
        args.logger.info('finish get resume east')

    args.logger.info('finish get network')

    ds, data_size = create_east_dataset(img_root=args.data_root, txt_root=args.txt_root, batch_size=args.per_batch_size,
                                        device_num=args.group_size, rank=args.rank, is_training=True)
    args.logger.info('Finish loading dataset')

    args.steps_per_epoch = int(
        data_size /
        args.per_batch_size /
        args.group_size)

    if not args.ckpt_interval:
        args.ckpt_interval = args.steps_per_epoch

    # get learnning rate
    lr = get_lr(args)
    opt = Adam(
        params=get_param_groups(network),
        learning_rate=Tensor(
            lr,
            ms.float32))
    loss_scale = FixedLossScaleManager(1.0, drop_overflow_update=True)
    model = Model(network, optimizer=opt, loss_scale_manager=loss_scale)
    network.set_train()

    # save the network model and parameters for subsequence fine-tuning
    config_ck = CheckpointConfig(
        save_checkpoint_steps=100,
        keep_checkpoint_max=1)
    # group layers into an object with training and evaluation features
    save_ckpt_path = os.path.join(
        args.outputs_dir, 'ckpt_' + str(args.rank) + '/')
    ckpoint_cb = ModelCheckpoint(
        prefix="checkpoint_east",
        directory=save_ckpt_path,
        config=config_ck)
    callback = []
    if args.rank == 0:
        callback = [
            TimeMonitor(
                data_size=data_size),
            LossMonitor(),
            ckpoint_cb]

    save_ckpt_path = os.path.join(
        args.outputs_dir, 'ckpt_' + str(args.rank) + '/')
    model.train(
        args.max_epoch,
        ds,
        callbacks=callback,
        dataset_sink_mode=False)
    args.logger.info('==========end training===============')
