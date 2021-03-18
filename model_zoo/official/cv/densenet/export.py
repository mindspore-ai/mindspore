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
"""export checkpoint file into air, onnx, mindir models"""
import argparse
import numpy as np

from mindspore.common import dtype as mstype
from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

parser = argparse.ArgumentParser(description="densenet export")

parser.add_argument("--net", type=str, default='', help="Densenet Model, densenet100 or densenet121")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="densenet", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if args.net == "densenet100":
    from src.config import config_100 as config
    from src.network.densenet import DenseNet100 as DenseNet
else:
    from src.config import config_121 as config
    from src.network.densenet import DenseNet121 as DenseNet

if __name__ == "__main__":
    network = DenseNet(config.num_classes)

    param_dict = load_checkpoint(args.ckpt_file)

    param_dict_new = {}
    for key, value in param_dict.items():
        if key.startswith("moments."):
            continue
        elif key.startswith("network."):
            param_dict_new[key[8:]] = value
        else:
            param_dict_new[key] = value

    load_param_into_net(network, param_dict_new)

    network.add_flags_recursive(fp16=True)
    network.set_train(False)

    shape = [int(args.batch_size), 3] + [int(config.image_size.split(",")[0]), int(config.image_size.split(",")[1])]
    input_data = Tensor(np.zeros(shape), mstype.float32)

    export(network, input_data, file_name=args.file_name, file_format=args.file_format)
