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
"""
mobilenetv2 export file.
"""
import argparse
import numpy as np
from mindspore import Tensor, export, context
from src.config import set_config
from src.models import define_net, load_ckpt
from src.utils import set_context

parser = argparse.ArgumentParser(description="mobilenetv2 export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="mobilenetv2", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument('--platform', type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"),
                    help='run platform, only support GPU, CPU and Ascend')
args = parser.parse_args()
args.is_training = False
args.run_distribute = False

context.set_context(mode=context.GRAPH_MODE, device_target=args.platform)
if args.platform == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    cfg = set_config(args)
    set_context(cfg)
    _, _, net = define_net(cfg, args.is_training)

    load_ckpt(net, args.ckpt_file)
    input_shp = [args.batch_size, 3, cfg.image_height, cfg.image_width]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    export(net, input_array, file_name=args.file_name, file_format=args.file_format)
