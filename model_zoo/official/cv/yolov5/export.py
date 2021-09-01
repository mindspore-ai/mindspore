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
import argparse
import numpy as np

import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from src.yolo import YOLOV5s_Infer

parser = argparse.ArgumentParser(description='yolov5 export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument('--yolov5_version', default='yolov5s', type=str,
                    help='The version of YOLOv5, options: yolov5s, yolov5m, yolov5l, yolov5x')

parser.add_argument("--testing_shape", type=int, default=640, help="test shape")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="yolov5", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"],
                    default="Ascend", help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == "__main__":
    ts_shape = args.testing_shape // 2

    dict_version = {'yolov5s': 0, 'yolov5m': 1, 'yolov5l': 2, 'yolov5x': 3}
    args.file_name = args.file_name + '_' + args.yolov5_version

    network = YOLOV5s_Infer(args.testing_shape, version=dict_version[args.yolov5_version])
    network.set_train(False)

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)

    input_data = Tensor(np.zeros([args.batch_size, 12, ts_shape, ts_shape]), mindspore.float32)

    export(network, input_data, file_name=args.file_name, file_format=args.file_format)
    print('==========success export===============')
