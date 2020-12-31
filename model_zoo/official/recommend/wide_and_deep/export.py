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
##############export checkpoint file into air, mindir and onnx models#################
"""
import argparse
import numpy as np

from mindspore import Tensor, context, load_checkpoint, export, load_param_into_net

from eval import ModelBuilder
from src.config import WideDeepConfig

parser = argparse.ArgumentParser(description="wide_and_deep export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="wide_and_deep", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    widedeep_config = WideDeepConfig()
    widedeep_config.argparse_init()

    net_builder = ModelBuilder()
    _, eval_net = net_builder.get_net(widedeep_config)

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)

    ids = Tensor(np.ones([widedeep_config.eval_batch_size, widedeep_config.field_size]).astype(np.int32))
    wts = Tensor(np.ones([widedeep_config.eval_batch_size, widedeep_config.field_size]).astype(np.float32))
    label = Tensor(np.ones([widedeep_config.eval_batch_size, 1]).astype(np.float32))
    input_tensor_list = [ids, wts, label]
    export(eval_net, *input_tensor_list, file_name=args.file_name, file_format=args.file_format)
