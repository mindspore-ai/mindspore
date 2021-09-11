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
"""train FCN8s."""


from mindspore import context, Tensor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from src.data import dataset as data_generator
from src.loss import loss
from src.utils.lr_scheduler import CosineAnnealingLR
from src.nets.FCN8s import FCN8s
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num


set_seed(1)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def train():
    device_num = get_device_num()
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True, save_graphs=False,
                        device_target=config.device_target, device_id=get_device_id())
    # init multicards training
    config.rank = 0
    config.group_size = 1
    if device_num > 1:
        parallel_mode = ParallelMode.DATA_PARALLEL
        if config.parallel_mode in ParallelMode.MODE_LIST:
            parallel_mode = config.parallel_mode
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=device_num)
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()

    # dataset
    dataset = data_generator.SegDataset(image_mean=config.image_mean,
                                        image_std=config.image_std,
                                        data_file=config.data_file,
                                        batch_size=config.train_batch_size,
                                        crop_size=config.crop_size,
                                        max_scale=config.max_scale,
                                        min_scale=config.min_scale,
                                        ignore_label=config.ignore_label,
                                        num_classes=config.num_classes,
                                        num_readers=2,
                                        num_parallel_calls=4,
                                        shard_id=config.rank,
                                        shard_num=config.group_size)
    dataset = dataset.get_dataset(repeat=1)

    net = FCN8s(n_class=config.num_classes)
    if context.get_auto_parallel_context("parallel_mode") in [ParallelMode.SEMI_AUTO_PARALLEL,
                                                              ParallelMode.AUTO_PARALLEL]:
        net.set_model_parallel_shard_strategy(device_num)
    loss_ = loss.SoftmaxCrossEntropyLoss(config.num_classes, config.ignore_label, device_num=device_num)

    # load pretrained vgg16 parameters to init FCN8s
    if config.ckpt_vgg16:
        param_vgg = load_checkpoint(config.ckpt_vgg16)
        is_model_zoo_vgg16 = "layers.0.weight" in param_vgg
        if is_model_zoo_vgg16:
            _is_bn_used = "layers.1.gamma" in param_vgg
            _is_imagenet_used = param_vgg['classifier.6.bias'].shape == (1000,)
            if not _is_bn_used or not _is_imagenet_used:
                raise Exception("Please use the vgg16 checkpoint which use BN and trained by ImageNet dataset.")
        idx = 0
        param_dict = {}
        for layer_id in range(1, 6):
            sub_layer_num = 2 if layer_id < 3 else 3
            for sub_layer_id in range(sub_layer_num):
                # conv param
                y_weight = 'conv{}.{}.weight'.format(layer_id, 3 * sub_layer_id)
                x_weight = 'vgg16_feature_extractor.conv{}_{}.0.weight'.format(layer_id, sub_layer_id + 1)
                if is_model_zoo_vgg16:
                    x_weight = 'layers.{}.weight'.format(idx)
                param_dict[y_weight] = param_vgg[x_weight]
                # BatchNorm param
                y_gamma = 'conv{}.{}.gamma'.format(layer_id, 3 * sub_layer_id + 1)
                y_beta = 'conv{}.{}.beta'.format(layer_id, 3 * sub_layer_id + 1)
                x_gamma = 'vgg16_feature_extractor.conv{}_{}.1.gamma'.format(layer_id, sub_layer_id + 1)
                x_beta = 'vgg16_feature_extractor.conv{}_{}.1.beta'.format(layer_id, sub_layer_id + 1)
                if is_model_zoo_vgg16:
                    x_gamma = 'layers.{}.gamma'.format(idx + 1)
                    x_beta = 'layers.{}.beta'.format(idx + 1)
                param_dict[y_gamma] = param_vgg[x_gamma]
                param_dict[y_beta] = param_vgg[x_beta]
                idx += 3
            idx += 1
        load_param_into_net(net, param_dict)
    # load pretrained FCN8s
    elif config.ckpt_pre_trained:
        param_dict = load_checkpoint(config.ckpt_pre_trained)
        load_param_into_net(net, param_dict)

    # optimizer
    iters_per_epoch = dataset.get_dataset_size()

    lr_scheduler = CosineAnnealingLR(config.base_lr,
                                     config.train_epochs,
                                     iters_per_epoch,
                                     config.train_epochs,
                                     warmup_epochs=0,
                                     eta_min=0)
    lr = Tensor(lr_scheduler.get_lr())

    # loss scale
    manager_loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    if config.device_target == "Ascend":
        optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.0001,
                                loss_scale=config.loss_scale)
        model = Model(net, loss_fn=loss_, loss_scale_manager=manager_loss_scale, optimizer=optimizer, amp_level="O3")
    elif config.device_target == "GPU":
        optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.0001)
        model = Model(net, loss_fn=loss_, optimizer=optimizer)
    else:
        raise ValueError("Unsupported platform.")

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    if config.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_steps,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=config.model, directory=config.ckpt_dir, config=config_ck)
        cbs.append(ckpoint_cb)

    model.train(config.train_epochs, dataset, callbacks=cbs)


if __name__ == '__main__':
    train()
