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
"""
eval.
"""
import os
import argparse
import numpy as np

from mindspore import context, Tensor
from mindspore import nn
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype

from src.pet_dataset import create_dataset
from src.config import cfg
from src.resnet_imgnet import resnet50


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str,
                    default='resnet50-imgnet-0.65x-80.24.ckpt', help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str,
                    default='/home/hankai/xiaoan/data/test.mindrecord', help='Dataset path')
parser.add_argument('--platform', type=str, default='GPU', help='run platform')
args_opt = parser.parse_args()


if __name__ == '__main__':
    config_platform = cfg
    if args_opt.platform == "Ascend":
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                            device_id=device_id, save_graphs=False)
    elif args_opt.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU", save_graphs=False)
    else:
        raise ValueError("Unsupported platform.")

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    dataset = create_dataset(dataset_path=args_opt.dataset_path,
                             do_train=False,
                             config=config_platform,
                             platform=args_opt.platform,
                             batch_size=config_platform.batch_size)
    step_size = dataset.get_dataset_size()

    index = []
    with open('index.txt', 'r') as f:
        for line in f:
            ind = Tensor((np.array(line.strip('\n').split(' ')[:-1])).astype(np.int32).reshape(-1, 1))
            index.append(ind)

    net = resnet50(
        rate=0.65, class_num=config_platform.num_classes, index=index)

    if args_opt.platform == "Ascend":
        net.to_float(mstype.float16)
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.to_float(mstype.float32)

    if args_opt.checkpoint_path:
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        load_param_into_net(net, param_dict)

    net.set_train(False)

    model = Model(net, loss_fn=loss, metrics={'acc'})
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
