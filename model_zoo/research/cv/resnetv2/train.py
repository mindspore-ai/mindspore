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
""" train.py """
import os
import argparse

from mindspore.nn import Momentum
from mindspore.context import ParallelMode
from mindspore import context, Model, load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore.common import set_seed
from mindspore.common.tensor import Tensor
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from src.lr_generator import get_lr


parser = argparse.ArgumentParser(description='Image classification.')
parser.add_argument('--net', type=str, default='resnetv2_50',
                    help='Resnetv2 Model, resnetv2_50, resnetv2_101, resnetv2_152')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Dataset, cifar10, cifar100, imagenet2012')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--dataset_path', type=str, default="../cifar-10/cifar-10-batches-bin",
                    help='Dataset path.')
args_opt = parser.parse_args()

# import net
if args_opt.net == "resnetv2_50":
    from src.resnetv2 import PreActResNet50 as resnetv2
elif args_opt.net == 'resnetv2_101':
    from src.resnetv2 import PreActResNet101 as resnetv2
elif args_opt.net == 'resnetv2_152':
    from src.resnetv2 import PreActResNet152 as resnetv2

# import dataset
if args_opt.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
elif args_opt.dataset == "cifar100":
    from src.dataset import create_dataset2 as create_dataset
elif args_opt.dataset == 'imagenet2012':
    raise ValueError("ImageNet2012 dataset not yet supported")

# import config
if args_opt.net == "resnetv2_50" or args_opt.net == "resnetv2_101" or args_opt.net == "resnetv2_152":
    if args_opt.dataset == "cifar10":
        from src.config import config1 as config
    elif args_opt.dataset == 'cifar100':
        from src.config import config2 as config
    elif args_opt.dataset == 'imagenet2012':
        raise ValueError("ImageNet2012 dataset not yet supported")

set_seed(1)

if __name__ == '__main__':
    print("============== Starting Training ==============")
    target = args_opt.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args_opt.run_distribute:
        init()
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            # init parallel training parameters
            context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            # init HCCL
            init()
        else:
            raise ValueError("run distribute for GPU not yet supported")
    else:
        try:
            device_id = int(os.getenv('DEVICE_ID'))
        except TypeError:
            device_id = 0

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path)
    step_size = dataset.get_dataset_size()

    # define net
    epoch_size = config.epoch_size
    net = resnetv2(config.class_num)

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)

    # init lr
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define loss, opt, model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                   config.weight_decay, config.loss_scale)
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()

    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_save_dir = config.save_checkpoint_path
    ckpoint_cb = ModelCheckpoint(prefix=f"train_{args_opt.net}_{args_opt.dataset}",
                                 directory=ckpt_save_dir, config=config_ck)

    # train
    if args_opt.run_distribute:
        callbacks = [time_cb, loss_cb]
        if device_id == 0:
            callbacks = [time_cb, loss_cb, ckpoint_cb]
    else:
        callbacks = [time_cb, loss_cb, ckpoint_cb]

    model.train(epoch_size, dataset, callbacks=callbacks)
