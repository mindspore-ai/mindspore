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
"""export file"""
import argparse
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from src.efficientnet import efficientnet_b0
from src.config import dataset_config

parser = argparse.ArgumentParser(description="efficientnet export")
parser.add_argument("--width", type=int, default=224, help="input width")
parser.add_argument("--height", type=int, default=224, help="input height")
parser.add_argument('--dataset', type=str, default='ImageNet', choices=['ImageNet', 'CIFAR10'],
                    help='ImageNet or CIFAR10')
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="efficientnet", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"],
                    default="MINDIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["GPU", "CPU"], default="GPU",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)


if __name__ == "__main__":
    if args.device_target not in ("GPU", "CPU"):
        raise ValueError("Only supported CPU and GPU now.")

    dataset_type = args.dataset.lower()
    cfg = dataset_config[dataset_type].cfg

    net = efficientnet_b0(num_classes=cfg.num_classes,
                          drop_rate=cfg.drop,
                          drop_connect_rate=cfg.drop_connect,
                          global_pool=cfg.gp,
                          bn_tf=cfg.bn_tf,
                          )

    ckpt = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, ckpt)
    net.set_train(False)

    image = Tensor(np.ones([cfg.batch_size, 3, args.height, args.width], np.float32))
    export(net, image, file_name=args.file_name, file_format=args.file_format)
