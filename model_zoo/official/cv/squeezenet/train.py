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
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth

set_seed(1)

if config.net_name == "squeezenet":
    from src.squeezenet import SqueezeNet as squeezenet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.dataset import create_dataset_imagenet as create_dataset
else:
    from src.squeezenet import SqueezeNet_Residual as squeezenet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.dataset import create_dataset_imagenet as create_dataset

@moxing_wrapper()
def train_net():
    """train net"""
    target = config.device_target
    ckpt_save_dir = config.output_path

    # init context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target)
    device_num = 1
    if config.run_distribute:
        if target == "Ascend":
            device_id = get_device_id()
            device_num = config.device_num
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
            init()
        # GPU target
        else:
            init()
            device_num = get_group_size()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
        ckpt_save_dir = ckpt_save_dir + "/ckpt_" + str(
            get_rank()) + "/"

    # create dataset
    dataset = create_dataset(dataset_path=config.data_path,
                             do_train=True,
                             repeat_num=1,
                             batch_size=config.batch_size,
                             target=target)
    step_size = dataset.get_dataset_size()

    # define net
    net = squeezenet(num_classes=config.class_num)

    # load checkpoint
    if config.pre_trained:
        param_dict = load_checkpoint(config.pre_trained)
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
    if config.dataset == "imagenet":
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
        ckpt_cb = ModelCheckpoint(prefix=config.net_name + '_' + config.dataset,
                                  directory=ckpt_save_dir,
                                  config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size - config.pretrain_epoch_size,
                dataset,
                callbacks=cb)

if __name__ == '__main__':
    train_net()
