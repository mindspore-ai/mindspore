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

"""Export file"""

import argparse
import numpy as np

from mindspore import dtype as mstype
from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from src.retinahead import retinahead
from src.backbone import resnet152
from src.config import config

parser = argparse.ArgumentParser(description="retinanet_resnet152 export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="retinanet_resnet152", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == "__main__":
    network = retinahead(backbone=resnet152(80), config=config, is_training=False)

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)

    network.set_train(False)

    shape = [args.batch_size, 3] + [640, 640]
    input_data = Tensor(np.zeros(shape), mstype.float32)

    export(network, input_data, file_name=args.file_name, file_format=args.file_format)
