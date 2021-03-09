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
"""train squeezenet."""
import os
import argparse
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--net', type=str, default='squeezenet', choices=['squeezenet', 'squeezenet_residual'],
                    help='Model.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'], help='Dataset.')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
args_opt = parser.parse_args()

set_seed(1)

if args_opt.net == "squeezenet":
    from src.squeezenet import SqueezeNet as squeezenet
    if args_opt.dataset == "cifar10":
        from src.config import config1 as config
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.config import config2 as config
        from src.dataset import create_dataset_imagenet as create_dataset
else:
    from src.squeezenet import SqueezeNet_Residual as squeezenet
    if args_opt.dataset == "cifar10":
        from src.config import config3 as config
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.config import config4 as config
        from src.dataset import create_dataset_imagenet as create_dataset

if __name__ == '__main__':
    target = args_opt.device_target
    ckpt_save_dir = config.save_checkpoint_path

    # init context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target)
    if args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id,
                                enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(
                device_num=args_opt.device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
            init()
        # GPU target
        else:
            print("Squeezenet training on GPU performs badly now, and it is still in research..."
                  "See model_zoo/research/cv/squeezenet to get up-to-date details.")
            init()
            context.set_auto_parallel_context(
                device_num=get_group_size(),
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
        ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(
            get_rank()) + "/"

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path,
                             do_train=True,
                             repeat_num=1,
                             batch_size=config.batch_size,
                             target=target)
    step_size = dataset.get_dataset_size()

    # define net
    net = squeezenet(num_classes=config.class_num)

    # load checkpoint
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)

    # init lr
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                total_epochs=config.epoch_size,
                warmup_epochs=config.warmup_epochs,
                pretrain_epochs=config.pretrain_epoch_size,
                steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define loss
    if args_opt.dataset == "imagenet":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True,
                                  reduction='mean',
                                  smooth_factor=config.label_smooth_factor,
                                  num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define opt, model
    if target == "Ascend":
        loss_scale = FixedLossScaleManager(config.loss_scale,
                                           drop_overflow_update=False)
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                       lr,
                       config.momentum,
                       config.weight_decay,
                       config.loss_scale,
                       use_nesterov=True)
        model = Model(net,
                      loss_fn=loss,
                      optimizer=opt,
                      loss_scale_manager=loss_scale,
                      metrics={'acc'},
                      amp_level="O2",
                      keep_batchnorm_fp32=False)
    else:
        if target == "GPU":
            # GPU target
            print("Squeezenet training on GPU performs badly now, and it is still in research..."
                  "See model_zoo/research/cv/squeezenet to get up-to-date details.")
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                       lr,
                       config.momentum,
                       config.weight_decay,
                       use_nesterov=True)
        model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
            keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=args_opt.net + '_' + args_opt.dataset,
                                  directory=ckpt_save_dir,
                                  config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size - config.pretrain_epoch_size,
                dataset,
                callbacks=cb)
