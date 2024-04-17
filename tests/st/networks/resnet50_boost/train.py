# Copyright 2022 Huawei Technologies Co., Ltd
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
"""train resnet."""
import datetime
import glob
import os
import time
import numpy as np

import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim import Momentum, LARS
from mindspore.context import ParallelMode
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.amp import build_train_network
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
import mindspore.log as logger

from src.lr_generator import get_lr, warmup_cosine_annealing_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_num
from src.resnet import conv_variance_scaling_initializer
from src.resnet import resnet50 as resnet


set_seed(1)


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def set_graph_kernel_context(run_platform, net_name):
    if run_platform == "GPU" and net_name == "resnet101":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=Conv2D")


def set_parameter():
    """set_parameter"""
    target = config.device_target
    if target == "CPU":
        config.run_distribute = False

    # init context
    if config.mode_name == 'GRAPH':
        if target == "Ascend":
            rank_save_graphs_path = os.path.join(config.save_graphs_path, "soma", str(os.getenv('DEVICE_ID')))
            context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=config.save_graphs,
                                save_graphs_path=rank_save_graphs_path)
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=config.save_graphs)
        set_graph_kernel_context(target, config.net_name)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)

    if config.parameter_server:
        context.set_ps_context(enable_ps=True)
    if config.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            if config.net_name == "resnet50" or config.net_name == "se-resnet50":
                if config.boost_mode not in ["O1", "O2"]:
                    context.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
            elif config.net_name in ["resnet101", "resnet152"]:
                context.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
            init()
        # GPU target
        else:
            init()
            context.set_auto_parallel_context(device_num=get_device_num(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            if config.net_name == "resnet50":
                context.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)


def load_pre_trained_checkpoint():
    """
    Load checkpoint according to pre_trained path.
    """
    param_dict = None
    if config.pre_trained:
        if os.path.isdir(config.pre_trained):
            ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path, "ckpt_0")
            ckpt_pattern = os.path.join(ckpt_save_dir, "*.ckpt")
            ckpt_files = glob.glob(ckpt_pattern)
            if not ckpt_files:
                logger.warning(f"There is no ckpt file in {ckpt_save_dir}, "
                               f"pre_trained is unsupported.")
            else:
                ckpt_files.sort(key=os.path.getmtime, reverse=True)
                time_stamp = datetime.datetime.now()
                print(f"time stamp {time_stamp.strftime('%Y.%m.%d-%H:%M:%S')}"
                      f" pre trained ckpt model {ckpt_files[0]} loading",
                      flush=True)
                param_dict = load_checkpoint(ckpt_files[0])
        elif os.path.isfile(config.pre_trained):
            param_dict = load_checkpoint(config.pre_trained)
        else:
            print(f"Invalid pre_trained {config.pre_trained} parameter.")
    return param_dict


def init_weight(net, param_dict):
    """init_weight"""
    if config.pre_trained:
        if param_dict:
            if param_dict.get("epoch_num") and param_dict.get("step_num"):
                config.has_trained_epoch = int(param_dict["epoch_num"].data.asnumpy())
                config.has_trained_step = int(param_dict["step_num"].data.asnumpy())
            else:
                config.has_trained_epoch = 0
                config.has_trained_step = 0

            if config.filter_weight:
                filter_list = [x.name for x in net.end_point.get_parameters()]
                filter_checkpoint_parameter_by_list(param_dict, filter_list)
            load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if config.conv_init == "XavierUniform":
                    cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                                 cell.weight.shape,
                                                                 cell.weight.dtype))
                elif config.conv_init == "TruncatedNormal":
                    weight = conv_variance_scaling_initializer(cell.in_channels,
                                                               cell.out_channels,
                                                               cell.kernel_size[0])
                    cell.weight.set_data(weight)
            if isinstance(cell, nn.Dense):
                if config.dense_init == "TruncatedNormal":
                    cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                                 cell.weight.shape,
                                                                 cell.weight.dtype))
                elif config.dense_init == "RandomNormal":
                    in_channel = cell.in_channels
                    out_channel = cell.out_channels
                    weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
                    weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=cell.weight.dtype)
                    cell.weight.set_data(weight)


def init_lr(step_size):
    """init lr"""
    if config.optimizer == "Thor":
        from src.lr_generator import get_thor_lr
        lr = get_thor_lr(0, config.lr_init, config.lr_decay, config.lr_end_epoch, step_size, decay_epochs=39)
    else:
        if config.net_name in ("resnet18", "resnet34", "resnet50", "resnet152", "se-resnet50"):
            lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                        warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                        lr_decay_mode=config.lr_decay_mode)
        else:
            lr = warmup_cosine_annealing_lr(config.lr, step_size, config.warmup_epochs, config.epoch_size,
                                            config.pretrain_epoch_size * step_size)
    return lr


def init_loss_scale():
    if config.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    return loss


def init_group_params(net):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


@moxing_wrapper()
def train_net():
    """train net"""
    set_parameter()
    ckpt_param_dict = load_pre_trained_checkpoint()
    net = resnet(class_num=config.class_num)
    if config.parameter_server:
        net.set_param_ps()

    init_weight(net=net, param_dict=ckpt_param_dict)
    lr = Tensor(init_lr(step_size=500))
    # define opt
    group_params = init_group_params(net)
    opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    if config.optimizer == "LARS":
        opt = LARS(opt, epsilon=config.lars_epsilon, coefficient=config.lars_coefficient,
                   lars_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'bias' not in x.name)
    loss = init_loss_scale()
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    train_network = build_train_network(net, opt, loss, level="O2", boost_level=config.boost_mode,
                                        loss_scale_manager=loss_scale, keep_batchnorm_fp32=False)
    for _ in range(500):
        image = Tensor(np.random.rand(32, 3, 224, 224), dtype=mindspore.float32)
        label = Tensor(np.random.randint(0, 10, [32]), dtype=mindspore.int32)
        begin_time = time.time()
        _ = train_network(image, label)
        print("train one step cost time: {} ms".format((time.time() - begin_time) * 1000))

if __name__ == '__main__':
    train_net()
