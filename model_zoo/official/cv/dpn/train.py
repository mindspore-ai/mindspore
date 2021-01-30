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
"""DPN model train with MindSpore"""
import os
import argparse

from mindspore import context
from mindspore import Tensor
from mindspore.nn import SGD
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.imagenet_dataset import classification_dataset
from src.dpn import dpns
from src.config import config
from src.lr_scheduler import get_lr_drop, get_lr_warmup
from src.crossentropy import CrossEntropy
from src.callbacks import SaveCallback

device_id = int(os.getenv('DEVICE_ID'))

set_seed(1)


def parse_args():
    """parameters"""
    parser = argparse.ArgumentParser('dpn training')

    # dataset related
    parser.add_argument('--data_dir', type=str, default='', help='Imagenet data dir')
    # network related
    parser.add_argument('--pretrained', default='', type=str, help='ckpt path to load')
    # distributed related
    parser.add_argument('--is_distributed', type=int, default=1, help='if multi device')
    parser.add_argument('--ckpt_path', type=str, default='', help='ckpt path to save')
    parser.add_argument('--eval_each_epoch', type=int, default=0, help='evaluate on each epoch')
    args, _ = parser.parse_known_args()
    args.image_size = config.image_size
    args.num_classes = config.num_classes
    args.lr_init = config.lr_init
    args.lr_max = config.lr_max
    args.factor = config.factor
    args.global_step = config.global_step
    args.epoch_number_to_drop = config.epoch_number_to_drop
    args.epoch_size = config.epoch_size
    args.warmup_epochs = config.warmup_epochs
    args.weight_decay = config.weight_decay
    args.momentum = config.momentum
    args.batch_size = config.batch_size
    args.num_parallel_workers = config.num_parallel_workers
    args.backbone = config.backbone
    args.loss_scale_num = config.loss_scale_num
    args.is_save_on_master = config.is_save_on_master
    args.rank = config.rank
    args.group_size = config.group_size
    args.dataset = config.dataset
    args.label_smooth = config.label_smooth
    args.label_smooth_factor = config.label_smooth_factor
    args.keep_checkpoint_max = config.keep_checkpoint_max
    args.lr_schedule = config.lr_schedule
    return args


def dpn_train(args):
    # init context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", save_graphs=False, device_id=device_id)
    # init distributed
    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
        context.set_auto_parallel_context(device_num=args.group_size, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1
    # create dataset
    args.train_dir = os.path.join(args.data_dir, 'train')
    args.eval_dir = os.path.join(args.data_dir, 'val')
    train_dataset = classification_dataset(args.train_dir,
                                           image_size=args.image_size,
                                           per_batch_size=args.batch_size,
                                           max_epoch=1,
                                           num_parallel_workers=args.num_parallel_workers,
                                           shuffle=True,
                                           rank=args.rank,
                                           group_size=args.group_size)
    if args.eval_each_epoch:
        print("create eval_dataset")
        eval_dataset = classification_dataset(args.eval_dir,
                                              image_size=args.image_size,
                                              per_batch_size=args.batch_size,
                                              max_epoch=1,
                                              num_parallel_workers=args.num_parallel_workers,
                                              shuffle=False,
                                              rank=args.rank,
                                              group_size=args.group_size,
                                              mode='eval')
    train_step_size = train_dataset.get_dataset_size()

    # choose net
    net = dpns[args.backbone](num_classes=args.num_classes)

    # load checkpoint
    if os.path.isfile(args.pretrained):
        print("load ckpt")
        load_param_into_net(net, load_checkpoint(args.pretrained))
    # learing rate schedule
    if args.lr_schedule == 'drop':
        print("lr_schedule:drop")
        lr = Tensor(get_lr_drop(global_step=args.global_step,
                                total_epochs=args.epoch_size,
                                steps_per_epoch=train_step_size,
                                lr_init=args.lr_init,
                                factor=args.factor))
    elif args.lr_schedule == 'warmup':
        print("lr_schedule:warmup")
        lr = Tensor(get_lr_warmup(global_step=args.global_step,
                                  total_epochs=args.epoch_size,
                                  steps_per_epoch=train_step_size,
                                  lr_init=args.lr_init,
                                  lr_max=args.lr_max,
                                  warmup_epochs=args.warmup_epochs))

    # optimizer
    opt = SGD(net.trainable_params(),
              lr,
              momentum=args.momentum,
              weight_decay=args.weight_decay,
              loss_scale=args.loss_scale_num)
    # loss scale
    loss_scale = FixedLossScaleManager(args.loss_scale_num, False)
    # loss function
    if args.dataset == "imagenet-1K":
        print("Use SoftmaxCrossEntropyWithLogits")
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    else:
        if not args.label_smooth:
            args.label_smooth_factor = 0.0
        print("Use Label_smooth CrossEntropy")
        loss = CrossEntropy(smooth_factor=args.label_smooth_factor, num_classes=args.num_classes)
    # create model
    model = Model(net, amp_level="O2",
                  keep_batchnorm_fp32=False,
                  loss_fn=loss,
                  optimizer=opt,
                  loss_scale_manager=loss_scale,
                  metrics={'top_1_accuracy', 'top_5_accuracy'})

    # loss/time monitor & ckpt save callback
    loss_cb = LossMonitor()
    time_cb = TimeMonitor(data_size=train_step_size)
    cb = [loss_cb, time_cb]
    if args.rank_save_ckpt_flag:
        if args.eval_each_epoch:
            save_cb = SaveCallback(model, eval_dataset, args.ckpt_path)
            cb += [save_cb]
        else:
            config_ck = CheckpointConfig(save_checkpoint_steps=train_step_size,
                                         keep_checkpoint_max=args.keep_checkpoint_max)
            ckpoint_cb = ModelCheckpoint(prefix="dpn", directory=args.ckpt_path, config=config_ck)
            cb.append(ckpoint_cb)
    # train model
    model.train(args.epoch_size, train_dataset, callbacks=cb)


if __name__ == '__main__':
    dpn_train(parse_args())
    print('DPN training success!')
