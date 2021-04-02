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
"""
######################## train SimCLR example ########################
train simclr and get network model files(.ckpt) :
python train.py --train_dataset_path /YourDataPath
"""
import ast
import argparse
import os
from src.nt_xent import NT_Xent_Loss
from src.optimizer import get_train_optimizer as get_optimizer
from src.dataset import create_dataset
from src.simclr_model import SimCLR
from src.resnet import resnet50 as resnet
from mindspore import nn
from mindspore import context
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import initializer as weight_init
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net

parser = argparse.ArgumentParser(description='MindSpore SimCLR')
parser.add_argument('--device_target', type=str, default='Ascend',
                    help='Device target, Currently only Ascend is supported.')
parser.add_argument('--run_cloudbrain', type=ast.literal_eval, default=True,
                    help='Whether it is running on CloudBrain platform.')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=True, help='Run distributed training.')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument("--device_id", type=int, default=0, help="device id, default is 0.")
parser.add_argument('--dataset_name', type=str, default="cifar10", help='Dataset, Currently only cifar10 is supported.')
parser.add_argument('--train_url', default=None, help='Cloudbrain Location of training outputs.\
                    This parameter needs to be set when running on the cloud brain platform.')
parser.add_argument('--data_url', default=None, help='Cloudbrain Location of data.\
                    This parameter needs to be set when running on the cloud brain platform.')
parser.add_argument('--train_dataset_path', type=str, default="./cifar/train",
                    help='Dataset path for training classifier. '
                         'This parameter needs to be set when running on the host.')
parser.add_argument('--train_output_path', type=str, default="./outputs", help='Location of ckpt and log.\
                    This parameter needs to be set when running on the host.')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size, default is 128.')
parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training, default is 200.')
parser.add_argument('--projection_dimension', type=int, default=128,
                    help='Projection output dimensionality, default is 128.')
parser.add_argument('--width_multiplier', type=int, default=1, help='width_multiplier for ResNet50')
parser.add_argument("--temperature", type=float, default=0.5, help="temperature for loss")
parser.add_argument('--pre_trained_path', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument("--pretrain_epoch_size", type=int, default=0,
                    help="real_epoch_size = epoch_size - pretrain_epoch_size.")
parser.add_argument("--save_checkpoint_epochs", type=int, default=1, help="Save checkpoint epochs, default is 1.")
parser.add_argument('--save_graphs', type=ast.literal_eval, default=False,
                    help='whether save graphs, default is False.')
parser.add_argument('--optimizer', type=str, default="Adam", help='Optimizer, Currently only Adam is supported.')
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--warmup_epochs", type=int, default=15, help="warmup epochs.")
parser.add_argument('--use_crop', type=ast.literal_eval, default=True, help='RandomResizedCrop')
parser.add_argument('--use_flip', type=ast.literal_eval, default=True, help='RandomHorizontalFlip')
parser.add_argument('--use_color_jitter', type=ast.literal_eval, default=True, help='RandomColorAdjust')
parser.add_argument('--use_color_gray', type=ast.literal_eval, default=True, help='RandomGrayscale')
parser.add_argument('--use_blur', type=ast.literal_eval, default=False, help='GaussianBlur')
parser.add_argument('--use_norm', type=ast.literal_eval, default=False, help='Normalize')

args = parser.parse_args()
local_data_url = './cache/data'
local_train_url = './cache/train'
_local_train_url = local_train_url

if args.device_target != "Ascend":
    raise ValueError("Unsupported device target.")
if args.run_distribute:
    args.device_id = int(os.getenv("DEVICE_ID"))
    if args.device_num > int(os.getenv("RANK_SIZE")) or args.device_num == 1:
        args.device_num = int(os.getenv("RANK_SIZE"))
    context.set_context(device_id=args.device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=args.save_graphs)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True, device_num=args.device_num)
    init()
    args.rank = get_rank()
    local_data_url = os.path.join(local_data_url, str(args.device_id))
    local_train_url = os.path.join(local_train_url, str(args.device_id))
    args.train_output_path = os.path.join(args.train_output_path, str(args.device_id))
else:
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        save_graphs=args.save_graphs, device_id=args.device_id)
    args.rank = 0
    args.device_num = 1

if args.run_cloudbrain:
    import moxing as mox
    args.train_dataset_path = os.path.join(local_data_url, "train")
    args.train_output_path = local_train_url
    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)

set_seed(1)

class NetWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(NetWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data_x, data_y, label):
        _, _, x_pred, y_pred = self._backbone(data_x, data_y)
        return self._loss_fn(x_pred, y_pred)

if __name__ == "__main__":
    dataset = create_dataset(args, dataset_mode="train_endcoder")
    # Net.
    base_net = resnet(1, args.width_multiplier, cifar_stem=args.dataset_name == "cifar10")
    net = SimCLR(base_net, args.projection_dimension, base_net.end_point.in_channels)
    # init weight
    if args.pre_trained_path:
        if args.run_cloudbrain:
            mox.file.copy_parallel(src_url=args.pre_trained_path, dst_url=local_data_url+'/pre_train.ckpt')
            param_dict = load_checkpoint(local_data_url+'/pre_train.ckpt')
        else:
            param_dict = load_checkpoint(args.pre_trained_path)
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
    optimizer = get_optimizer(net, dataset.get_dataset_size(), args)
    loss = NT_Xent_Loss(args.batch_size, args.temperature)
    net_loss = NetWithLossCell(net, loss)
    train_net = nn.TrainOneStepCell(net_loss, optimizer)
    model = Model(train_net)
    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_epochs)
    ckpts_dir = os.path.join(args.train_output_path, "checkpoint")
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_simclr", directory=ckpts_dir, config=config_ck)
    print("============== Starting Training ==============")
    model.train(args.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, LossMonitor()])
    if args.device_id == 0:
        mox.file.copy_parallel(src_url=_local_train_url, dst_url=args.train_url)
