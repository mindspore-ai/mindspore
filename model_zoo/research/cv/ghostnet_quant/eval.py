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
from mindspore import context
from mindspore import nn
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
from src.dataset import create_dataset
from src.config import config_ascend, config_gpu
from src.ghostnet import ghostnet_1x

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str,
                    default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str,
                    default=None, help='Dataset path')
parser.add_argument('--platform', type=str, default=None, help='run platform')
args_opt = parser.parse_args()


if __name__ == '__main__':
    config_platform = None
    if args_opt.platform == "Ascend":
        config_platform = config_ascend
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                            device_id=device_id, save_graphs=False)
    elif args_opt.platform == "GPU":
        config_platform = config_gpu
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU", save_graphs=False)
    else:
        raise ValueError("Unsupported platform.")

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net = ghostnet_1x(num_classes=config_platform.num_classes)

    if args_opt.platform == "Ascend":
        net.to_float(mstype.float16)
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.to_float(mstype.float32)

    dataset = create_dataset(dataset_path=args_opt.dataset_path,
                             do_train=False,
                             config=config_platform,
                             platform=args_opt.platform,
                             batch_size=config_platform.batch_size)
    step_size = dataset.get_dataset_size()

    if args_opt.checkpoint_path:
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        load_param_into_net(net, param_dict)
    net.set_train(False)

    model = Model(net, loss_fn=loss, metrics={'acc'})
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
