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
"""Train mobilenetV2 on ImageNet"""

import os
import argparse
import random
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore import nn
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint
from mindspore.communication.management import init
from mindspore.train.quant import quant
import mindspore.dataset.engine as de

from src.dataset import create_dataset
from src.lr_generator import get_lr
from src.utils import Monitor, CrossEntropyWithLabelSmooth, _load_param_into_net
from src.config import config_ascend, config_ascend_quant
from src.mobilenetV2 import mobilenetV2

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--pre_trained', type=str, default=None, help='Pertained checkpoint path')
parser.add_argument('--device_target', type=str, default=None, help='Run device target')
parser.add_argument('--quantization_aware', type=bool, default=False, help='Use quantization aware training')
args_opt = parser.parse_args()

if args_opt.device_target == "Ascend":
    device_id = int(os.getenv('DEVICE_ID'))
    rank_id = int(os.getenv('RANK_ID'))
    rank_size = int(os.getenv('RANK_SIZE'))
    run_distribute = rank_size > 1
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id, save_graphs=False)
else:
    raise ValueError("Unsupported device target.")

if __name__ == '__main__':
    # train on ascend
    config = config_ascend_quant if args_opt.quantization_aware else config_ascend
    print("training args: {}".format(args_opt))
    print("training configure: {}".format(config))
    print("parallel args: rank_id {}, device_id {}, rank_size {}".format(rank_id, device_id, rank_size))
    epoch_size = config.epoch_size

    # distribute init
    if run_distribute:
        context.set_auto_parallel_context(device_num=rank_size,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True,
                                          mirror_mean=True)
        init()

    # define network
    network = mobilenetV2(num_classes=config.num_classes)
    # define loss
    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    # define dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path,
                             do_train=True,
                             config=config,
                             device_target=args_opt.device_target,
                             repeat_num=1,
                             batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()
    # load pre trained ckpt
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        _load_param_into_net(network, param_dict)

    # convert fusion network to quantization aware network
    if config.quantization_aware:
        network = quant.convert_quant_network(network,
                                              bn_fold=True,
                                              per_channel=[True, False],
                                              symmetric=[True, False])

    # get learning rate
    lr = Tensor(get_lr(global_step=config.start_epoch * step_size,
                       lr_init=0,
                       lr_end=0,
                       lr_max=config.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=epoch_size + config.start_epoch,
                       steps_per_epoch=step_size))

    # define optimization
    opt = nn.Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), lr, config.momentum,
                      config.weight_decay)
    # define model
    model = Model(network, loss_fn=loss, optimizer=opt)

    print("============== Starting Training ==============")
    callback = None
    if rank_id == 0:
        callback = [Monitor(lr_init=lr.asnumpy())]
        if config.save_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="mobilenetV2",
                                      directory=config.save_checkpoint_path,
                                      config=config_ck)
            callback += [ckpt_cb]
    model.train(epoch_size, dataset, callbacks=callback)
    print("============== End Training ==============")
