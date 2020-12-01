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
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore.compression.quant import QuantizationAwareTraining

from src.config import config_quant
from modelsresnet_quant_manual import resnet50_quant

parser = argparse.ArgumentParser(description='resnet50_quant export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--img_size", type=int, default=224, help="image size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="resnet50_quant", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)

if __name__ == "__main__":
    config = config_quant

    network = resnet50_quant(class_num=config.class_num)

    quantizer = QuantizationAwareTraining(bn_fold=True, per_channel=[True, False], symmetric=[True, False])
    network = quantizer.quantize(network)

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)

    network.set_train(False)
    shape = [config.batch_size, 3] + [args.img_size, args.img_size]
    input_data = Tensor(np.zeros(shape).astype(np.float32))

    export(network, input_data, file_name=args.file_name, file_format=args.file_format)
