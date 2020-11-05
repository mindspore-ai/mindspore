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
"""Training Interface"""
import sys
import os
import argparse
import copy

from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import ParallelMode, Model
from mindspore.train.callback import TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn import SGD, RMSProp, Loss, Top1CategoricalAccuracy, \
    Top5CategoricalAccuracy
from mindspore import context, Tensor

from src.dataset import create_dataset, create_dataset_val
from src.utils import add_weight_decay, count_params, str2bool, get_lr
from src.callback import EmaEvalCallBack, LossMonitor
from src.loss import LabelSmoothingCrossEntropy
from src.tinynet import tinynet

parser = argparse.ArgumentParser(description='Training')

# training parameters
parser.add_argument('--data_path', type=str, default="", metavar="DIR",
                    help='path to dataset')
parser.add_argument('--model', default='tinynet_c', type=str, metavar='MODEL',
                    help='Name of model to train (default: "tinynet_c"')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--drop', type=float, default=0.0, metavar='DROP',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=0.0, metavar='DROP',
                    help='Drop connect rate (default: 0.)')
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--ema-decay', type=float, default=0,
                    help='decay factor for model weights moving average \
                    (default: 0.999)')
parser.add_argument('--amp_level', type=str, default='O0')
parser.add_argument('--per_print_times', type=int, default=100)

# batch norm parameters
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that \
                    support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')

# parallel parameters
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--distributed', action='store_true', default=False)
parser.add_argument('--dataset_sink', action='store_true', default=True)

# checkpoint config
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--ckpt_save_epoch', type=int, default=1)
parser.add_argument('--loss_scale', type=int,
                    default=1024, help='static loss scale')
parser.add_argument('--train', type=str2bool, default=1, help='train or eval')
parser.add_argument('--GPU', action='store_true', default=False,
                    help='Use GPU for training (default: False)')


def main():
    """Main entrance for training"""
    args = parser.parse_args()
    print(sys.argv)
    devid, args.rank_id, args.rank_size = 0, 0, 1

    context.set_context(mode=context.GRAPH_MODE)

    if args.distributed:
        if args.GPU:
            init("nccl")
            context.set_context(device_target='GPU')
        else:
            init()
            devid = int(os.getenv('DEVICE_ID'))
            context.set_context(device_target='Ascend',
                                device_id=devid,
                                reserve_class_name_in_scope=False)
        context.reset_auto_parallel_context()
        args.rank_id = get_rank()
        args.rank_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          device_num=args.rank_size)
    else:
        if args.GPU:
            context.set_context(device_target='GPU')

    is_master = not args.distributed or (args.rank_id == 0)

    # parse model argument
    assert args.model.startswith(
        "tinynet"), "Only Tinynet models are supported."
    _, sub_name = args.model.split("_")
    net = tinynet(sub_model=sub_name,
                  num_classes=args.num_classes,
                  drop_rate=args.drop,
                  drop_connect_rate=args.drop_connect,
                  global_pool="avg",
                  bn_tf=args.bn_tf,
                  bn_momentum=args.bn_momentum,
                  bn_eps=args.bn_eps)

    if is_master:
        print("Total number of parameters:", count_params(net))
    # input image size of the network
    input_size = net.default_cfg['input_size'][1]

    train_dataset = val_dataset = None
    train_data_url = os.path.join(args.data_path, 'train')
    val_data_url = os.path.join(args.data_path, 'val')
    val_dataset = create_dataset_val(args.batch_size,
                                     val_data_url,
                                     workers=args.workers,
                                     distributed=False,
                                     input_size=input_size)

    if args.train:
        train_dataset = create_dataset(args.batch_size,
                                       train_data_url,
                                       workers=args.workers,
                                       distributed=args.distributed,
                                       input_size=input_size)
        batches_per_epoch = train_dataset.get_dataset_size()

    loss = LabelSmoothingCrossEntropy(
        smooth_factor=args.smoothing, num_classes=args.num_classes)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    loss_scale_manager = FixedLossScaleManager(
        args.loss_scale, drop_overflow_update=False)

    lr_array = get_lr(base_lr=args.lr,
                      total_epochs=args.epochs,
                      steps_per_epoch=batches_per_epoch,
                      decay_epochs=args.decay_epochs,
                      decay_rate=args.decay_rate,
                      warmup_epochs=args.warmup_epochs,
                      warmup_lr_init=args.warmup_lr,
                      global_epoch=0)
    lr = Tensor(lr_array)

    loss_cb = LossMonitor(lr_array,
                          args.epochs,
                          per_print_times=args.per_print_times,
                          start_epoch=0)

    param_group = add_weight_decay(net, weight_decay=args.weight_decay)

    if args.opt == 'sgd':
        if is_master:
            print('Using SGD optimizer')
        optimizer = SGD(param_group,
                        learning_rate=lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        loss_scale=args.loss_scale)

    elif args.opt == 'rmsprop':
        if is_master:
            print('Using rmsprop optimizer')
        optimizer = RMSProp(param_group,
                            learning_rate=lr,
                            decay=0.9,
                            weight_decay=args.weight_decay,
                            momentum=args.momentum,
                            epsilon=args.opt_eps,
                            loss_scale=args.loss_scale)

    loss.add_flags_recursive(fp32=True, fp16=False)
    eval_metrics = {'Validation-Loss': Loss(),
                    'Top1-Acc': Top1CategoricalAccuracy(),
                    'Top5-Acc': Top5CategoricalAccuracy()}

    if args.ckpt:
        ckpt = load_checkpoint(args.ckpt)
        load_param_into_net(net, ckpt)
        net.set_train(False)

    model = Model(net, loss, optimizer, metrics=eval_metrics,
                  loss_scale_manager=loss_scale_manager,
                  amp_level=args.amp_level)

    net_ema = copy.deepcopy(net)
    net_ema.set_train(False)
    assert args.ema_decay > 0, "EMA should be used in tinynet training."

    ema_cb = EmaEvalCallBack(network=net,
                             ema_network=net_ema,
                             loss_fn=loss,
                             eval_dataset=val_dataset,
                             decay=args.ema_decay,
                             save_epoch=args.ckpt_save_epoch,
                             dataset_sink_mode=args.dataset_sink,
                             start_epoch=0)

    callbacks = [loss_cb, ema_cb, time_cb] if is_master else []

    if is_master:
        print("Training on " + args.model
              + " with " + str(args.num_classes) + " classes")

    model.train(args.epochs, train_dataset, callbacks=callbacks,
                dataset_sink_mode=args.dataset_sink)


if __name__ == '__main__':
    main()
