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
"""train squeezenet."""
import ast
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
from mindspore.common import set_seed
from mindspore.nn.metrics import Accuracy
from mindspore.communication.management import init, get_rank
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.squeezenet import SqueezeNet as squeezenet
from src.config import config
from src.dataset import create_dataset_imagenet as create_dataset

parser = argparse.ArgumentParser(description='SqueezeNet1_1')
parser.add_argument('--net', type=str, default='squeezenet', help='Model.')
parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset.')
parser.add_argument('--run_cloudbrain', type=ast.literal_eval, default=False,
                    help='Whether it is running on CloudBrain platform.')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--data_url', type=str, default="None", help='Datapath')
parser.add_argument('--train_url', type=str, default="None", help='Train output path')
args_opt = parser.parse_args()

local_data_url = '/cache/data'
local_train_url = '/cache/ckpt'

set_seed(1)

if __name__ == '__main__':

    target = args_opt.device_target
    if args_opt.device_target != "Ascend":
        raise ValueError("Unsupported device target.")

    ckpt_save_dir = config.save_checkpoint_path

    # init context
    if args_opt.run_distribute:
        device_num = int(os.getenv("RANK_SIZE"))
        device_id = int(os.getenv("DEVICE_ID"))
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=target)
        context.set_context(device_id=device_id)
        context.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True)
        init()
        local_data_url = os.path.join(local_data_url, str(device_id))
        ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(get_rank()) + "/"
    else:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=target)
        ckpt_save_dir = config.save_checkpoint_path + "ckpt_squeezenet/"
    # create dataset
    if args_opt.run_cloudbrain:
        import moxing as mox
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        dataset = create_dataset(dataset_path=local_data_url,
                                 do_train=True,
                                 repeat_num=1,
                                 batch_size=config.batch_size,
                                 target=target,
                                 run_distribute=args_opt.run_distribute)
    else:
        dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                 do_train=True,
                                 repeat_num=1,
                                 batch_size=config.batch_size,
                                 target=target,
                                 run_distribute=args_opt.run_distribute)
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
                  metrics={'acc': Accuracy()},
                  amp_level="O2",
                  keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
            keep_checkpoint_max=config.keep_checkpoint_max)
        if args_opt.run_cloudbrain:
            local_train_url = os.path.join(local_train_url, str(device_id))
            ckpt_cb = ModelCheckpoint(prefix=args_opt.net + '_' + args_opt.dataset,
                                      directory=local_train_url,
                                      config=config_ck)
        else:
            ckpt_cb = ModelCheckpoint(prefix=args_opt.net + '_' + args_opt.dataset,
                                      directory=ckpt_save_dir,
                                      config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size - config.pretrain_epoch_size,
                dataset,
                callbacks=cb)
    if args_opt.run_cloudbrain:
        mox.file.copy_parallel("/cache/ckpt", args_opt.train_url)
