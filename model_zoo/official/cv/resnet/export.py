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
"""
##############export checkpoint file into air and onnx models#################
python export.py
"""
import argparse
import numpy as np

from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

parser = argparse.ArgumentParser(description='resnet export')
parser.add_argument('--network_dataset', type=str, default='resnet50_cifar10', choices=['resnet18_cifar10',
                                                                                        'resnet18_imagenet2012',
                                                                                        'resnet50_cifar10',
                                                                                        'resnet50_imagenet2012',
                                                                                        'resnet101_imagenet2012',
                                                                                        "se-resnet50_imagenet2012"],
                    help='network and dataset name.')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="resnet", help="output file name.")
parser.add_argument('--width', type=int, default=224, help='input width')
parser.add_argument('--height', type=int, default=224, help='input height')
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target(default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':

    if args.network_dataset == 'resnet18_cifar10':
        from src.config import config1 as config
        from src.resnet import resnet18 as resnet
    elif args.network_dataset == 'resnet18_imagenet2012':
        from src.config import config2 as config
        from src.resnet import resnet18 as resnet
    elif args.network_dataset == 'resnet50_cifar10':
        from src.config import config1 as config
        from src.resnet import resnet50 as resnet
    elif args.network_dataset == 'resnet50_imagenet2012':
        from src.config import config2 as config
        from src.resnet import resnet50 as resnet
    elif args.network_dataset == 'resnet101_imagenet2012':
        from src.config import config3 as config
        from src.resnet import resnet101 as resnet
    elif args.network_dataset == 'se-resnet50_imagenet2012':
        from src.config import config4 as config
        from src.resnet import se_resnet50 as resnet
    else:
        raise ValueError("network and dataset is not support.")

    net = resnet(config.class_num)

    assert args.ckpt_file is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)
