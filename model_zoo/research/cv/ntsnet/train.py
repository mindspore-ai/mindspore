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
"""ntsnet train."""
import ast
import math
import os
import argparse
from mindspore.train.callback import CheckpointConfig, TimeMonitor
from mindspore import context, nn, Tensor, set_seed, Model
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from src.config import config
from src.dataset import create_dataset_train
from src.lr_generator import get_lr
from src.network import NTS_NET, WithLossCell, LossCallBack, ModelCheckpoint

parser = argparse.ArgumentParser(description='ntsnet train running')
parser.add_argument("--run_modelart", type=ast.literal_eval, default=False, help="Run on modelArt, default is false.")
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="Run distribute, default is false.")
parser.add_argument('--data_url', default=None,
                    help='Directory contains resnet50.ckpt and CUB_200_2011 dataset.')
parser.add_argument('--train_url', default=None, help='Directory of training output.')
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
args = parser.parse_args()
run_modelart = args.run_modelart
if run_modelart:
    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))
    local_input_url = '/cache/data' + str(device_id)
    local_output_url = '/cache/ckpt' + str(device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        save_graphs=False)
    context.set_context(device_id=device_id)
    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          global_rank=device_id,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        rank = get_rank()
    else:
        rank = 0
    import moxing as mox

    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_input_url)
elif args.run_distribute:
    device_id = args.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
    context.set_context(device_id=device_id)
    init()
    device_num = get_group_size()
    context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)
    local_input_url = args.data_url
    local_output_url = args.train_url
    rank = get_rank()
else:
    device_id = args.device_id
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
    context.set_context(device_id=device_id)
    rank = 0
    device_num = 1
    local_input_url = args.data_url
    local_output_url = args.train_url

learning_rate = config.learning_rate
momentum = config.momentum
weight_decay = config.weight_decay
batch_size = config.batch_size
num_train_images = config.num_train_images
num_epochs = config.num_epochs
steps_per_epoch = math.ceil(num_train_images / batch_size)
lr = Tensor(get_lr(global_step=0,
                   lr_init=0,
                   lr_max=learning_rate,
                   warmup_epochs=4,
                   total_epochs=num_epochs,
                   steps_per_epoch=steps_per_epoch))

if __name__ == '__main__':
    set_seed(1)
    resnet50Path = os.path.join(local_input_url, "resnet50.ckpt")
    ntsnet = NTS_NET(topK=6, resnet50Path=resnet50Path)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
    optimizer = nn.SGD(ntsnet.trainable_params(), learning_rate=lr, momentum=momentum, weight_decay=weight_decay)
    loss_net = WithLossCell(ntsnet, loss_fn)
    oneStepNTSNet = nn.TrainOneStepCell(loss_net, optimizer)

    train_data_set = create_dataset_train(train_path=os.path.join(local_input_url, "CUB_200_2011/train"),
                                          batch_size=batch_size)
    dataset_size = train_data_set.get_batch_size()
    time_cb = TimeMonitor(data_size=dataset_size)

    loss_cb = LossCallBack(rank_id=rank, local_output_url=local_output_url, device_num=device_num, device_id=device_id,
                           args=args, run_modelart=run_modelart)
    cb = [time_cb, loss_cb]

    if config.save_checkpoint and rank == 0:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * steps_per_epoch,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(local_output_url, "ckpt_" + str(rank) + "/")

        ckpoint_cb = ModelCheckpoint(prefix=config.prefix, directory=save_checkpoint_path, ckconfig=ckptconfig,
                                     device_num=device_num, device_id=device_id, args=args, run_modelart=run_modelart)
        cb += [ckpoint_cb]

    model = Model(oneStepNTSNet, amp_level="O3", keep_batchnorm_fp32=False)
    model.train(config.num_epochs, train_data_set, callbacks=cb, dataset_sink_mode=True)
