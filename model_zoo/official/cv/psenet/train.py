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


import argparse
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.dataset import train_dataset_creator
from src.config import config
from src.ETSNET.etsnet import ETSNet
from src.ETSNET.dice_loss import DiceLoss
from src.network_define import WithLossCell, TrainOneStepCell, LossCallBack
from src.lr_schedule import dynamic_lr

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--run_distribute', default=False, action='store_true',
                    help='Run distribute, default is false.')
parser.add_argument('--pre_trained', type=str, default='', help='Pretrain file path.')
parser.add_argument('--device_id', type=int, default=0, help='Device id, default is 0.')
parser.add_argument('--device_num', type=int, default=1, help='Use device nums, default is 1.')
args = parser.parse_args()

set_seed(1)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)

def train():
    rank_id = 0
    if args.run_distribute:
        context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
        rank_id = get_rank()

    # dataset/network/criterion/optim
    ds = train_dataset_creator(rank_id, args.device_num)
    step_size = ds.get_dataset_size()
    print('Create dataset done!')

    config.INFERENCE = False
    net = ETSNet(config)
    net = net.set_train()
    param_dict = load_checkpoint(args.pre_trained)
    load_param_into_net(net, param_dict)
    print('Load Pretrained parameters done!')

    criterion = DiceLoss(batch_size=config.TRAIN_BATCH_SIZE)

    lrs = dynamic_lr(config.BASE_LR, config.TRAIN_TOTAL_ITER, config.WARMUP_STEP, config.WARMUP_RATIO)
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lrs, momentum=0.99, weight_decay=5e-4)

    # warp model
    net = WithLossCell(net, criterion)
    if args.run_distribute:
        net = TrainOneStepCell(net, opt, reduce_flag=True, mean=True, degree=args.device_num)
    else:
        net = TrainOneStepCell(net, opt)

    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossCallBack(per_print_times=10)
    # set and apply parameters of check point config.TRAIN_MODEL_SAVE_PATH
    ckpoint_cf = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=2)
    ckpoint_cb = ModelCheckpoint(prefix="ETSNet", config=ckpoint_cf,
                                 directory="./ckpt_{}".format(rank_id))

    model = Model(net)
    model.train(config.TRAIN_REPEAT_NUM, ds, dataset_sink_mode=True, callbacks=[time_cb, loss_cb, ckpoint_cb])

if __name__ == '__main__':
    train()
