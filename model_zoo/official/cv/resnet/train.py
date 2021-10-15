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
import numpy as np
from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim import Momentum, thor, LARS
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.train_thor import ConvertModelUtils
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
import mindspore.log as logger
from src.lr_generator import get_lr, warmup_cosine_annealing_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.eval_callback import EvalCallBack
from src.metric import DistAccuracy, ClassifyCorrectCell
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_rank_id, get_device_num
from src.resnet import conv_variance_scaling_initializer

set_seed(1)

if config.net_name in ("resnet18", "resnet34", "resnet50"):
    if config.net_name == "resnet18":
        from src.resnet import resnet18 as resnet
    if config.net_name == "resnet34":
        from src.resnet import resnet34 as resnet
    if config.net_name == "resnet50":
        from src.resnet import resnet50 as resnet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset1 as create_dataset
    else:
        if config.mode_name == "GRAPH":
            from src.dataset import create_dataset2 as create_dataset
        else:
            from src.dataset import create_dataset_pynative as create_dataset
elif config.net_name == "resnet101":
    from src.resnet import resnet101 as resnet
    from src.dataset import create_dataset3 as create_dataset
else:
    from src.resnet import se_resnet50 as resnet
    from src.dataset import create_dataset4 as create_dataset


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    return res[metrics_name]

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
        context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
        set_graph_kernel_context(target, config.net_name)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
    if config.parameter_server:
        context.set_ps_context(enable_ps=True)
    if config.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            set_algo_parameters(elementwise_op_strategy_follow=True)
            if config.net_name == "resnet50" or config.net_name == "se-resnet50":
                if config.boost_mode not in ["O1", "O2"]:
                    context.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
            elif config.net_name == "resnet101":
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

def init_weight(net):
    """init_weight"""
    if config.pre_trained:
        param_dict = load_checkpoint(config.pre_trained)
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
        if config.net_name in ("resnet18", "resnet34", "resnet50", "se-resnet50"):
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

def run_eval(target, model, ckpt_save_dir, cb):
    """run_eval"""
    if config.run_eval:
        if config.eval_dataset_path is None or (not os.path.isdir(config.eval_dataset_path)):
            raise ValueError("{} is not a existing path.".format(config.eval_dataset_path))
        eval_dataset = create_dataset(dataset_path=config.eval_dataset_path, do_train=False,
                                      batch_size=config.batch_size, target=target, enable_cache=config.enable_cache,
                                      cache_session_id=config.cache_session_id)
        eval_param_dict = {"model": model, "dataset": eval_dataset, "metrics_name": "acc"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=config.save_best_ckpt,
                               ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_acc.ckpt",
                               metrics_name="acc")
        cb += [eval_cb]


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    if config.enable_modelarts and config.run_distribute:
        ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank_id()) + "/"
    else:
        if config.run_distribute:
            ckpt_save_dir = ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    return ckpt_save_dir


@moxing_wrapper()
def train_net():
    """train net"""
    target = config.device_target
    set_parameter()
    dataset = create_dataset(dataset_path=config.data_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target,
                             distribute=config.run_distribute)
    step_size = dataset.get_dataset_size()
    net = resnet(class_num=config.class_num)
    if config.parameter_server:
        net.set_param_ps()
    init_weight(net=net)
    lr = Tensor(init_lr(step_size=step_size))
    # define opt
    group_params = init_group_params(net)
    opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    if config.optimizer == "LARS":
        opt = LARS(opt, epsilon=config.lars_epsilon, coefficient=config.lars_coefficient,
                   lars_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'bias' not in x.name)
    loss = init_loss_scale()
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    dist_eval_network = ClassifyCorrectCell(net) if config.run_distribute else None
    metrics = {"acc"}
    if config.run_distribute:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size, device_num=config.device_num)}
    if (config.net_name not in ("resnet18", "resnet34", "resnet50", "resnet101", "se-resnet50")) or \
        config.parameter_server or target == "CPU":
        ## fp32 training
        model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics, eval_network=dist_eval_network)
    else:
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                      amp_level="O2", boost_level=config.boost_mode, keep_batchnorm_fp32=False,
                      eval_network=dist_eval_network)

    if config.optimizer == "Thor" and config.dataset == "imagenet2012":
        from src.lr_generator import get_thor_damping
        damping = get_thor_damping(0, config.damping_init, config.damping_decay, 70, step_size)
        split_indices = [26, 53]
        opt = thor(net, lr, Tensor(damping), config.momentum, config.weight_decay, config.loss_scale,
                   config.batch_size, split_indices=split_indices, frequency=config.frequency)
        model = ConvertModelUtils().convert_to_thor_model(model=model, network=net, loss_fn=loss, optimizer=opt,
                                                          loss_scale_manager=loss_scale, metrics={'acc'},
                                                          amp_level="O2", keep_batchnorm_fp32=False)
        config.run_eval = False
        logger.warning("Thor optimizer not support evaluation while training.")

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    ckpt_save_dir = set_save_ckpt_dir()
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]
    run_eval(target, model, ckpt_save_dir, cb)
    # train model
    if config.net_name == "se-resnet50":
        config.epoch_size = config.train_epoch_size
    dataset_sink_mode = (not config.parameter_server) and target != "CPU"
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)

    if config.run_eval and config.enable_cache:
        print("Remember to shut down the cache server via \"cache_admin --stop\"")

if __name__ == '__main__':
    train_net()
