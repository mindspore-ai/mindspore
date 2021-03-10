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

"""Train retinanet and get checkpoint files."""

import os
import argparse
import ast
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, Callback
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.retinanet import  retinanetWithLossCell, TrainingWrapper, retinanet50, resnet50
from src.config import config
from src.dataset import create_retinanet_dataset
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter


set_seed(1)
class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("lr:[{:8.6f}]".format(self.lr_init[cb_params.cur_step_num-1]), flush=True)

def main():
    parser = argparse.ArgumentParser(description="retinanet training")

    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is False.")
    parser.add_argument("--workers", type=int, default=24, help="Num parallel workers.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate, default is 0.1.")
    parser.add_argument("--mode", type=str, default="sink", help="Run sink mode or not, default is sink.")
    parser.add_argument("--epoch_size", type=int, default=500, help="Epoch size, default is 500.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--pre_trained", type=str, default=None, help="Pretrained Checkpoint file path.")
    parser.add_argument("--pre_trained_epoch_size", type=int, default=0, help="Pretrained epoch size.")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=1, help="Save checkpoint epochs, default is 1.")
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter weight parameters, default is False.")
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend"),
                        help="run platform, only support Ascend.")
    args_opt = parser.parse_args()

    if args_opt.run_platform == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        if args_opt.distribute:
            if os.getenv("DEVICE_ID", "not_set").isdigit():
                context.set_context(device_id=int(os.getenv("DEVICE_ID")))
            init()
            device_num = args_opt.device_num
            rank = get_rank()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
        else:
            rank = 0
            device_num = 1
            context.set_context(device_id=args_opt.device_id)

    else:
        raise ValueError("Unsupported platform.")

    mindrecord_file = os.path.join(config.mindrecord_dir, "retinanet.mindrecord0")

    loss_scale = float(args_opt.loss_scale)

    # When create MindDataset, using the fitst mindrecord file, such as retinanet.mindrecord0.
    dataset = create_retinanet_dataset(mindrecord_file, repeat_num=1,
                                       num_parallel_workers=args_opt.workers,
                                       batch_size=args_opt.batch_size, device_num=device_num, rank=rank)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")


    backbone = resnet50(config.num_classes)
    retinanet = retinanet50(backbone, config)
    net = retinanetWithLossCell(retinanet, config)
    init_net_param(net)

    if args_opt.pre_trained:
        if args_opt.pre_trained_epoch_size <= 0:
            raise KeyError("pre_trained_epoch_size must be greater than 0.")
        param_dict = load_checkpoint(args_opt.pre_trained)
        if args_opt.filter_weight:
            filter_checkpoint_parameter(param_dict)
        load_param_into_net(net, param_dict)

    lr = Tensor(get_lr(global_step=config.global_step,
                       lr_init=config.lr_init, lr_end=config.lr_end_rate * args_opt.lr, lr_max=args_opt.lr,
                       warmup_epochs1=config.warmup_epochs1, warmup_epochs2=config.warmup_epochs2,
                       warmup_epochs3=config.warmup_epochs3, warmup_epochs4=config.warmup_epochs4,
                       warmup_epochs5=config.warmup_epochs5, total_epochs=args_opt.epoch_size,
                       steps_per_epoch=dataset_size))
    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                      config.momentum, config.weight_decay, loss_scale)
    net = TrainingWrapper(net, opt, loss_scale)
    model = Model(net)
    print("Start train retinanet, the first epoch will be slower because of the graph compilation.")
    cb = [TimeMonitor(), LossMonitor()]
    cb += [Monitor(lr_init=lr.asnumpy())]
    config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="retinanet", directory=config.save_checkpoint_path, config=config_ck)
    if args_opt.distribute:
        if rank == 0:
            cb += [ckpt_cb]
        model.train(args_opt.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
    else:
        cb += [ckpt_cb]
        model.train(args_opt.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)

if __name__ == '__main__':
    main()
