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
"""export checkpoint file into air, onnx, mindir models"""
import argparse
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
import src.config as cfg

parser = argparse.ArgumentParser(description='fasterrcnn_export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--file_name", type=str, default="faster_rcnn", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
parser.add_argument('--ckpt_file', type=str, default='', help='fasterrcnn ckpt file.')
parser.add_argument("--backbone", type=str, required=True, \
                    help="backbone network name, options:resnet50v1.0, resnet50v1.5, resnet101 ,resnet152")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if args.backbone in ("resnet50v1.5", "resnet101", "resnet152"):
    from src.FasterRcnn.faster_rcnn_resnet import FasterRcnn_Infer
    if args.backbone == "resnet50v1.5":
        config = cfg.get_config("./src/config_50.yaml")
    elif args.backbone == "resnet101":
        config = cfg.get_config("./src/config_101.yaml")
    elif args.backbone == "resnet152":
        config = cfg.get_config("./src/config_152.yaml")

elif args.backbone == "resnet50v1.0":
    config = cfg.get_config("./src/config_50.yaml")
    from src.FasterRcnn.faster_rcnn_resnet50v1 import FasterRcnn_Infer

if __name__ == '__main__':
    net = FasterRcnn_Infer(config=config)

    param_dict = load_checkpoint(args.ckpt_file)

    param_dict_new = {}
    for key, value in param_dict.items():
        param_dict_new["network." + key] = value

    load_param_into_net(net, param_dict_new)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    img = Tensor(np.zeros([config.test_batch_size, 3, config.img_height, config.img_width]), mstype.float32)
    img_metas = Tensor(np.random.uniform(0.0, 1.0, size=[config.test_batch_size, 4]), mstype.float32)

    export(net, img, img_metas, file_name=args.file_name, file_format=args.file_format)
