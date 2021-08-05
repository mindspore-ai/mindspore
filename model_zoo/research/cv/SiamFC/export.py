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
"""export checkpoint file into models"""
import argparse
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, export, load_param_into_net
from src.alexnet import SiameseAlexNet
parser = argparse.ArgumentParser(description='siamfc export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument('--model_path', default='/root/HRBEU-MedAI/SiamFC/models/siamfc_{}.ckpt/',
                    type=str, help='eval one special video')
parser.add_argument('--file_name', type=str, default='/root/HRBEU-MedAI/SiamFC/models',
                    help='SiamFc output file name.')
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR',
                    help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == "__main__":
    net = SiameseAlexNet(train=False)
    load_param_into_net(net, load_checkpoint(args.model_path), strict_load=True)
    net.set_train(False)

    input_data_exemplar = Tensor(np.zeros([3, 256, 6, 6]), ms.float32)
    input_data_instance = Tensor(np.zeros([3, 3, 255, 255]), ms.float32)
    export(net, input_data_exemplar, input_data_instance, file_name=args.file_name,
           file_format=args.file_format)
