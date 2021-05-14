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
"""train"""
import os
import math
import mindspore.dataset as ds
from mindspore import Parameter, set_seed, context
from mindspore.context import ParallelMode
from mindspore.common.initializer import initializer, HeUniform, XavierUniform, Uniform, Normal, Zero
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.args import args
from src.data.bicubic import bicubic
from src.data.imagenet import ImgData
from src.ipt_model import IPT
from src.utils import Trainer


def _calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = shape[2] * shape[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    :param net: network to be initialized
    :type net: nn.Module
    :param init_type: the name of an initialization method: normal | xavier | kaiming | orthogonal
    :type init_type: str
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    :type init_gain: float
    """

    for _, cell in net.cells_and_names():
        classname = cell.__class__.__name__
        if hasattr(cell, 'in_proj_layer'):
            cell.in_proj_layer = Parameter(initializer(HeUniform(negative_slope=math.sqrt(5)), cell.in_proj_layer.shape,
                                                       cell.in_proj_layer.dtype), name=cell.in_proj_layer.name)
        if hasattr(cell, 'weight'):
            if init_type == 'normal':
                cell.weight = Parameter(initializer(Normal(init_gain), cell.weight.shape,
                                                    cell.weight.dtype), name=cell.weight.name)
            elif init_type == 'xavier':
                cell.weight = Parameter(initializer(XavierUniform(init_gain), cell.weight.shape,
                                                    cell.weight.dtype), name=cell.weight.name)
            elif init_type == "he":
                cell.weight = Parameter(initializer(HeUniform(negative_slope=math.sqrt(5)), cell.weight.shape,
                                                    cell.weight.dtype), name=cell.weight.name)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(cell, 'bias') and cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                bound = 1 / math.sqrt(fan_in)
                cell.bias = Parameter(initializer(Uniform(bound), cell.bias.shape, cell.bias.dtype),
                                      name=cell.bias.name)
        elif classname.find('BatchNorm2d') != -1:
            cell.gamma = Parameter(initializer(Normal(1.0), cell.gamma.default_input.shape()), name=cell.gamma.name)
            cell.beta = Parameter(initializer(Zero(), cell.beta.default_input.shape()), name=cell.beta.name)

    print('initialize network weight with %s' % init_type)

def train_net(distribute, imagenet, epochs):
    """Train net"""
    set_seed(1)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)

    if imagenet == 1:
        train_dataset = ImgData(args)
    else:
        train_dataset = data.Data(args).loader_train

    if distribute:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=args.group_size, gradients_mean=True)
        print('Rank {}, args.group_size {}'.format(args.rank, args.group_size))
        if imagenet == 1:
            train_de_dataset = ds.GeneratorDataset(train_dataset,
                                                   ["HR", "Rain", "LRx2", "LRx3", "LRx4", "scales", "filename"],
                                                   num_shards=args.group_size, shard_id=args.rank, shuffle=True)
        else:
            train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR"], num_shards=args.group_size,
                                                   shard_id=args.rank, shuffle=True)
    else:
        if imagenet == 1:
            train_de_dataset = ds.GeneratorDataset(train_dataset,
                                                   ["HR", "Rain", "LRx2", "LRx3", "LRx4", "scales", "filename"],
                                                   shuffle=True)
        else:
            train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR"], shuffle=True)

    resize_fuc = bicubic()
    train_de_dataset = train_de_dataset.project(columns=["HR", "Rain", "LRx2", "LRx3", "LRx4", "filename"])
    train_de_dataset = train_de_dataset.batch(args.batch_size,
                                              input_columns=["HR", "Rain", "LRx2", "LRx3", "LRx4", "filename"],
                                              output_columns=["LR", "HR", "idx", "filename"],
                                              drop_remainder=True, per_batch_map=resize_fuc.forward)
    train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)

    net_work = IPT(args)
    init_weights(net_work, init_type='he', init_gain=1.0)
    print("Init net weight successfully")
    if args.pth_path:
        param_dict = load_checkpoint(args.pth_path)
        load_param_into_net(net_work, param_dict)
        print("Load net weight successfully")

    train_func = Trainer(args, train_loader, net_work)
    for epoch in range(0, epochs):
        train_func.update_learning_rate(epoch)
        train_func.train()

if __name__ == '__main__':
    train_net(distribute=args.distribute, imagenet=args.imagenet, epochs=args.epochs)
