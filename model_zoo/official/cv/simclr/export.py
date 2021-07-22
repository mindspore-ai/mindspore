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
"""
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, nn, load_checkpoint, load_param_into_net, export
from src.simclr_model import SimCLR, SimCLR_Classifier
from src.resnet import resnet50 as resnet

parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10'],
                    help='Dataset, Currently only cifar10 is supported.')
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend'],
                    help='Device target, Currently only Ascend is supported.')
parser.add_argument("--ckpt_simclr_encoder", type=str, required=True, help="Simclr encoder checkpoint file path.")
parser.add_argument("--ckpt_linear_classifier", type=str, required=True, help="Linear classifier checkpoint file path.")
parser.add_argument("--file_name", type=str, default="simclr_classifier", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR"], default="MINDIR", help="file format")
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
if args_opt.device_target == "Ascend":
    context.set_context(device_id=args_opt.device_id)


if __name__ == '__main__':
    if args_opt.dataset_name != 'cifar10':
        raise ValueError("dataset is not support.")
    width_multiplier = 1
    cifar_stem = True
    projection_dimension = 128
    class_num = 10
    image_height = 32
    image_width = 32

    encoder = resnet(1, width_multiplier=width_multiplier, cifar_stem=cifar_stem)
    classifier = nn.Dense(encoder.end_point.in_channels, class_num)

    simclr = SimCLR(encoder, projection_dimension, encoder.end_point.in_channels)
    param_simclr = load_checkpoint(args_opt.ckpt_simclr_encoder)
    load_param_into_net(simclr, param_simclr)

    param_classifier = load_checkpoint(args_opt.ckpt_linear_classifier)
    load_param_into_net(classifier, param_classifier)

    # export SimCLR_Classifier network
    simclr_classifier = SimCLR_Classifier(simclr.encoder, classifier)
    input_data = Tensor(np.zeros([args_opt.batch_size, 3, image_height, image_width]), mstype.float32)
    export(simclr_classifier, input_data, file_name=args_opt.file_name, file_format=args_opt.file_format)
