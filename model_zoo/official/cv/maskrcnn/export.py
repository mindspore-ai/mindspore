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
"""export checkpoint file into air, onnx, mindir models"""
import argparse
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.maskrcnn.mask_rcnn_r50 import MaskRcnn_Infer
from src.config import config

parser = argparse.ArgumentParser(description='maskrcnn export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="maskrcnn", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'GPU', 'CPU'], help='device target (default: Ascend)')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    net = MaskRcnn_Infer(config=config)
    param_dict = load_checkpoint(args.ckpt_file)

    param_dict_new = {}
    for key, value in param_dict.items():
        param_dict_new["network." + key] = value

    load_param_into_net(net, param_dict_new)
    net.set_train(False)

    bs = config.test_batch_size

    img = Tensor(np.zeros([args.batch_size, 3, config.img_height, config.img_width], np.float16))
    img_metas = Tensor(np.zeros([args.batch_size, 4], np.float16))

    input_data = [img, img_metas]
    export(net, *input_data, file_name=args.file_name, file_format=args.file_format)
