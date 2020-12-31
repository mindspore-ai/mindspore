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
"""export checkpoint file into models"""
import argparse
import numpy as np

from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, export

from src.vgg import vgg16

parser = argparse.ArgumentParser(description='VGG16 export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument('--dataset', type=str, choices=["cifar10", "imagenet2012"], default="cifar10", help='ckpt file')
parser.add_argument('--ckpt_file', type=str, required=True, help='vgg16 ckpt file.')
parser.add_argument('--file_name', type=str, default='vgg16', help='vgg16 output file name.')
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()

if args.dataset == "cifar10":
    from src.config import cifar_cfg as cfg
else:
    from src.config import imagenet_cfg as cfg

args.num_classes = cfg.num_classes
args.pad_mode = cfg.pad_mode
args.padding = cfg.padding
args.has_bias = cfg.has_bias
args.initialize_mode = cfg.initialize_mode
args.batch_norm = cfg.batch_norm
args.has_dropout = cfg.has_dropout
args.image_size = list(map(int, cfg.image_size.split(',')))

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    if args.dataset == "cifar10":
        net = vgg16(num_classes=args.num_classes, args=args)
    else:
        net = vgg16(args.num_classes, args, phase="test")
        net.add_flags_recursive(fp16=True)

    load_checkpoint(args.ckpt_file, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([cfg.batch_size, 3, args.image_size[0], args.image_size[1]]), mstype.float32)

    export(net, input_data, file_name=args.file_name, file_format=args.file_format)
