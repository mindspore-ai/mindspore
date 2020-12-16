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
"""Train mobilenetV2 on ImageNet."""

import os
import time
import random
import numpy as np

from mindspore import Tensor
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_rank
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import save_checkpoint
from mindspore.common import set_seed

from src.dataset import create_dataset, extract_features
from src.lr_generator import get_lr
from src.config import set_config

from src.args import train_parse_args
from src.utils import context_device_init, switch_precision, config_ckpoint
from src.models import CrossEntropyWithLabelSmooth, define_net, load_ckpt

set_seed(1)

if __name__ == '__main__':
    args_opt = train_parse_args()
    args_opt.dataset_path = os.path.abspath(args_opt.dataset_path)
    config = set_config(args_opt)
    start = time.time()

    print(f"train args: {args_opt}\ncfg: {config}")

    #set context and device init
    context_device_init(config)

    # define network
    backbone_net, head_net, net = define_net(config, args_opt.is_training)
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True, config=config)
    step_size = dataset.get_dataset_size()
    if args_opt.pretrain_ckpt:
        if args_opt.freeze_layer == "backbone":
            load_ckpt(backbone_net, args_opt.pretrain_ckpt, trainable=False)
            step_size = extract_features(backbone_net, args_opt.dataset_path, config)
        elif args_opt.filter_head:
            load_ckpt(backbone_net, args_opt.pretrain_ckpt)
        else:
            load_ckpt(net, args_opt.pretrain_ckpt)
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images' count of train dataset is more \
            than batch_size in config.py")

    # Currently, only Ascend support switch precision.
    switch_precision(net, mstype.float16, config)

    # define loss
    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(
            smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    epoch_size = config.epoch_size

    # get learning rate
    lr = Tensor(get_lr(global_step=0,
                       lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=epoch_size,
                       steps_per_epoch=step_size))

    if args_opt.pretrain_ckpt == "" or args_opt.freeze_layer != "backbone":
        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum, \
            config.weight_decay, config.loss_scale)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale)

        cb = config_ckpoint(config, lr, step_size)
        print("============== Starting Training ==============")
        model.train(epoch_size, dataset, callbacks=cb)
        print("============== End Training ==============")

    else:
        opt = Momentum(filter(lambda x: x.requires_grad, head_net.get_parameters()), lr, config.momentum, config.weight_decay)

        network = WithLossCell(head_net, loss)
        network = TrainOneStepCell(network, opt)
        network.set_train()

        features_path = args_opt.dataset_path + '_features'
        idx_list = list(range(step_size))
        rank = 0
        if config.run_distribute:
            rank = get_rank()
        save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
        if not os.path.isdir(save_ckpt_path):
            os.mkdir(save_ckpt_path)

        for epoch in range(epoch_size):
            random.shuffle(idx_list)
            epoch_start = time.time()
            losses = []
            for j in idx_list:
                feature = Tensor(np.load(os.path.join(features_path, f"feature_{j}.npy")))
                label = Tensor(np.load(os.path.join(features_path, f"label_{j}.npy")))
                losses.append(network(feature, label).asnumpy())
            epoch_mseconds = (time.time()-epoch_start) * 1000
            per_step_mseconds = epoch_mseconds / step_size
            print("epoch[{}/{}], iter[{}] cost: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}"\
            .format(epoch + 1, epoch_size, step_size, epoch_mseconds, per_step_mseconds, np.mean(np.array(losses))))
            if (epoch + 1) % config.save_checkpoint_epochs == 0:
                save_checkpoint(net, os.path.join(save_ckpt_path, f"mobilenetv2_{epoch+1}.ckpt"))
        print("total cost {:5.4f} s".format(time.time() - start))
