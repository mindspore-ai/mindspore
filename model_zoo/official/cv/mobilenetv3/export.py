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
mobilenetv3 export mindir.
"""
import argparse
import numpy as np
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export
from src.config import config_gpu
from src.config import config_cpu
from src.mobilenetV3 import mobilenet_v3_large


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Checkpoint file path')
parser.add_argument('--device_target', type=str, default="GPU", help='run device_target')
args_opt = parser.parse_args()

if __name__ == '__main__':
    cfg = None
    if args_opt.device_target == "GPU":
        cfg = config_gpu
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    elif args_opt.device_target == "CPU":
        cfg = config_cpu
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    else:
        raise ValueError("Unsupported device_target.")

    net = mobilenet_v3_large(num_classes=cfg.num_classes, activation="Softmax")

    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    input_shp = [1, 3, cfg.image_height, cfg.image_width]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    export(net, input_array, file_name=cfg.export_file, file_format=cfg.export_format)
