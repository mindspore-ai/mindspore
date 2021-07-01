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
# ===========================================================================
"""export ckpt to model"""

import argparse
import numpy as np

from mindspore import context, Tensor
from mindspore.common import set_seed
from mindspore.train.serialization import export, load_checkpoint
from src.config import ModelConfig
from src.fat_deepffm import ModelBuilder

parser = argparse.ArgumentParser(description="deepfm export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="deepfm", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
args = parser.parse_args()
set_seed(1)

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == "__main__":
    config = ModelConfig()

    model_builder = ModelBuilder(config)
    _, network = model_builder.get_train_eval_net()
    network.set_train(False)

    load_checkpoint(args.ckpt_file, net=network)

    batch_ids = Tensor(np.zeros([config.batch_size, config.cats_dim]).astype(np.int32))
    batch_wts = Tensor(np.zeros([config.batch_size, config.dense_dim]).astype(np.float32))
    labels = Tensor(np.zeros([config.batch_size, 1]).astype(np.float32))

    input_data = [batch_ids, batch_wts, labels]
    export(network, *input_data, file_name=args.file_name, file_format=args.file_format)
