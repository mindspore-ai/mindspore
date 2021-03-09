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

import argparse
import numpy as np

import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.ssd import SSD300, SsdInferWithDecoder, ssd_mobilenet_v2, ssd_mobilenet_v1_fpn, ssd_resnet50_fpn, ssd_vgg16
from src.config import config
from src.box_utils import default_boxes

parser = argparse.ArgumentParser(description='SSD export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="ssd", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "MINDIR"], default='AIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    if config.model == "ssd300":
        net = SSD300(ssd_mobilenet_v2(), config, is_training=False)
    elif config.model == "ssd_vgg16":
        net = ssd_vgg16(config=config)
    elif config.model == "ssd_mobilenet_v1_fpn":
        net = ssd_mobilenet_v1_fpn(config=config)
    elif config.model == "ssd_resnet50_fpn":
        net = ssd_resnet50_fpn(config=config)
    else:
        raise ValueError(f'config.model: {config.model} is not supported')
    net = SsdInferWithDecoder(net, Tensor(default_boxes), config)

    param_dict = load_checkpoint(args.ckpt_file)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)

    input_shp = [args.batch_size, 3] + config.img_shape
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp), mindspore.float32)
    export(net, input_array, file_name=args.file_name, file_format=args.file_format)
