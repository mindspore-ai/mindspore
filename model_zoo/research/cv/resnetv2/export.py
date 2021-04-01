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
"""Convert ckpt to air."""
import argparse
import numpy as np

from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

parser = argparse.ArgumentParser(description='resnet export')
parser.add_argument('--net', type=str, default='resnetv2_50',
                    help='Resnetv2 Model, resnetv2_50, resnetv2_101, resnetv2_152')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset, cifar10, cifar100, imagenet2012')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="resnetv2", help="output file name.")
parser.add_argument('--width', type=int, default=32, help='input width')
parser.add_argument('--height', type=int, default=32, help='input height')
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target(default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    # import net
    if args.net == "resnetv2_50":
        from src.resnetv2 import PreActResNet50 as resnetv2
    elif args.net == 'resnetv2_101':
        from src.resnetv2 import PreActResNet101 as resnetv2
    elif args.net == 'resnetv2_152':
        from src.resnetv2 import PreActResNet152 as resnetv2
    else:
        raise ValueError("network is not support.")

    # import config
    if args.net == "resnetv2_50" or args.net == "resnetv2_101" or args.net == "resnetv2_152":
        if args.dataset == "cifar10":
            from src.config import config1 as config
        elif args.dataset == 'cifar100':
            from src.config import config2 as config
        elif args.dataset == 'imagenet2012':
            raise ValueError("ImageNet2012 dataset not yet supported")
        else:
            raise ValueError("dataset is not support.")
    else:
        raise ValueError("network is not support.")

    net = resnetv2(config.class_num)

    assert args.ckpt_file is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)
