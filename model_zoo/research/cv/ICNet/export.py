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
import yaml
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
import mindspore.common.dtype as dtype
from src.models import ICNet

parser = argparse.ArgumentParser(description='maskrcnn export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="icnet", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'GPU', 'CPU'], help='device target (default: Ascend)')
parser.add_argument("--project_path", type=str, default='/root/ICNet/',
                    help="project_path,default is /root/ICNet/")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    config_path = args.project_path + "src/model_utils/icnet.yaml"
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
    net = ICNet()
    param_dict = load_checkpoint(args.ckpt_file)

    load_param_into_net(net, param_dict)
    net.set_train(False)

    img = Tensor(np.ones([args.batch_size, 3, cfg['model']["base_size"], cfg['model']["base_size"]*2]), dtype.float32)

    export(net, img, file_name=args.file_name, file_format=args.file_format)
