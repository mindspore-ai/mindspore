# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import os
import argparse
import ast
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim import Momentum, THOR
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.train_thor import ConvertModelUtils
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from src.lr_generator import get_lr, warmup_cosine_annealing_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.config import cfg

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--net', type=str, default=None, help='Resnet Model, resnet18, resnet50 or resnet101')
parser.add_argument('--dataset', type=str, default=None, help='Dataset, either cifar10 or imagenet2012')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')

parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--parameter_server', type=ast.literal_eval, default=False, help='Run parameter server train')
parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                    help="Filter head weight parameters, default is False.")
args_opt = parser.parse_args()

set_seed(1)

if args_opt.net in ("resnet18", "resnet50"):
    if args_opt.net == "resnet18":
        from src.resnet import resnet18 as resnet
    if args_opt.net == "resnet50":
        from src.resnet import resnet50 as resnet
    if args_opt.dataset == "cifar10":
        from src.config import config1 as config
        from src.dataset import create_dataset1 as create_dataset
    else:
        from src.config import config2 as config
        from src.dataset import create_dataset2 as create_dataset

elif args_opt.net == "resnet101":
    from src.resnet import resnet101 as resnet
    from src.config import config3 as config
    from src.dataset import create_dataset3 as create_dataset
else:
    from src.resnet import se_resnet50 as resnet
    from src.config import config4 as config
    from src.dataset import create_dataset4 as create_dataset

if cfg.optimizer == "Thor":
    if args_opt.device_target == "Ascend":
        from src.config import config_thor_Ascend as config
    else:
        from src.config import config_thor_gpu as config


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


if __name__ == '__main__':
    target = args_opt.device_target
    if target == "CPU":
        args_opt.run_distribute = False

    ckpt_save_dir = config.save_checkpoint_path

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args_opt.parameter_server:
        context.set_ps_context(enable_ps=True)
    if args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            if args_opt.net == "resnet50" or args_opt.net == "se-resnet50":
                context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
            elif args_opt.net == "resnet101":
                context.set_auto_parallel_context(all_reduce_fusion_config=[80, 210, 313])
            init()
        # GPU target
        else:
            init()
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            if args_opt.net == "resnet50":
                context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(get_rank()) + "/"

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target, distribute=args_opt.run_distribute)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet(class_num=config.class_num)
    if args_opt.parameter_server:
        net.set_param_ps()

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        if args_opt.filter_weight:
            filter_list = [x.name for x in net.end_point.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    # init lr
    if cfg.optimizer == "Thor":
        from src.lr_generator import get_thor_lr
        lr = get_thor_lr(0, config.lr_init, config.lr_decay, config.lr_end_epoch, step_size, decay_epochs=39)
    else:
        if args_opt.net in ("resnet18", "resnet50", "se-resnet50"):
            lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                        warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                        lr_decay_mode=config.lr_decay_mode)
        else:
            lr = warmup_cosine_annealing_lr(config.lr, step_size, config.warmup_epochs, config.epoch_size,
                                            config.pretrain_epoch_size * step_size)
    lr = Tensor(lr)

    # define opt
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
    opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    # define loss, model
    if target == "Ascend":
        if args_opt.dataset == "imagenet2012":
            if not config.use_label_smooth:
                config.label_smooth_factor = 0.0
            loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                      smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
        else:
            loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=False)
    else:
        # GPU and CPU target
        if args_opt.dataset == "imagenet2012":
            if not config.use_label_smooth:
                config.label_smooth_factor = 0.0
            loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                      smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
        else:
            loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

        if (args_opt.net == "resnet101" or args_opt.net == "resnet50") and \
            not args_opt.parameter_server and target != "CPU":
            opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum, config.weight_decay,
                           config.loss_scale)
            loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
            # Mixed precision
            model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                          amp_level="O2", keep_batchnorm_fp32=False)
        else:
            ## fp32 training
            opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum, config.weight_decay)
            model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})
    if cfg.optimizer == "Thor" and args_opt.dataset == "imagenet2012":
        from src.lr_generator import get_thor_damping
        damping = get_thor_damping(0, config.damping_init, config.damping_decay, 70, step_size)
        split_indices = [26, 53]
        opt = THOR(net, lr, Tensor(damping), config.momentum, config.weight_decay, config.loss_scale,
                   config.batch_size, split_indices=split_indices)
        model = ConvertModelUtils().convert_to_thor_model(model=model, network=net, loss_fn=loss, optimizer=opt,
                                                          loss_scale_manager=loss_scale, metrics={'acc'},
                                                          amp_level="O2", keep_batchnorm_fp32=False,
                                                          frequency=config.frequency)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    # train model
    if args_opt.net == "se-resnet50":
        config.epoch_size = config.train_epoch_size
    dataset_sink_mode = (not args_opt.parameter_server) and target != "CPU"
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)
