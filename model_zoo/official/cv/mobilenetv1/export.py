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

from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint

from src.mobilenet_v1 import mobilenet_v1 as mobilenet

parser = argparse.ArgumentParser(description="mobilenetv1 export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--dataset", type=str, default="imagenet2012", help="Dataset, either cifar10 or imagenet2012")
parser.add_argument('--width', type=int, default=224, help='input width')
parser.add_argument('--height', type=int, default=224, help='input height')
parser.add_argument("--file_name", type=str, default="mobilenetv1", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

if args.dataset == "cifar10":
    from src.config import config1 as config
else:
    from src.config import config2 as config

if __name__ == "__main__":
    target = args.device_target
    if target != "GPU":
        context.set_context(device_id=args.device_id)

    network = mobilenet(class_num=config.class_num)

    param_dict = load_checkpoint(args.ckpt_file, net=network)

    network.set_train(False)

    input_data = Tensor(np.zeros([config.batch_size, 3, args.height, args.width]).astype(np.float32))

    export(network, input_data, file_name=args.file_name, file_format=args.file_format)
