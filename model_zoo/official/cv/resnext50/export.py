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
resnext export mindir.
"""
import argparse
import numpy as np
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export
from src.config import config
from src.image_classification import get_network


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser('mindspore classification test')
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU'), help='run platform')

    parser.add_argument('--pretrained', type=str, required=True, help='fully path of pretrained model to load. '
                        'If it is a direction, it will test all ckpt')

    args, _ = parser.parse_known_args()
    args.image_size = config.image_size
    args.num_classes = config.num_classes
    args.backbone = config.backbone

    args.image_size = list(map(int, config.image_size.split(',')))
    args.image_height = args.image_size[0]
    args.image_width = args.image_size[1]
    args.export_format = config.export_format
    args.export_file = config.export_file
    return args

if __name__ == '__main__':
    args_export = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_export.platform)

    net = get_network(args_export.backbone, num_classes=args_export.num_classes, platform=args_export.platform)

    param_dict = load_checkpoint(args_export.pretrained)
    load_param_into_net(net, param_dict)
    input_shp = [1, 3, args_export.image_height, args_export.image_width]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    export(net, input_array, file_name=args_export.export_file, file_format=args_export.export_format)
