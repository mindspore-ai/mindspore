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
"""Export GENet_Res50 on ImageNet"""
import argparse
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from src.GENet import GE_resnet50 as net
from src.config import config1 as config

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--extra', type=str, default="True",
                    help='whether to use Depth-wise conv to down sample')
parser.add_argument('--mlp', type=str, default="True", help='bottleneck . whether to use 1*1 conv')
args_opt = parser.parse_args()

def trans_char_to_bool(str_):
    """
    Args:
        str_: string

    Returns:
        bool
    """
    result = False
    if str_.lower() == "true":
        result = True
    return result

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target,
                        save_graphs=False)
    # define fusion network
    mlp = trans_char_to_bool(args_opt.mlp)
    extra = trans_char_to_bool(args_opt.extra)
    network = net(class_num=config.class_num, extra=extra, mlp=mlp)

    # load checkpoint
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        not_load_param = load_param_into_net(network, param_dict)
        if not_load_param:
            raise ValueError("Load param into network fail!")
    # export network
    print("============== Starting export ==============")
    inputs = Tensor(np.ones([1, 3, 224, 224]))
    export(network, inputs, file_name="GENet_Res50")
    print("============== End export ==============")
