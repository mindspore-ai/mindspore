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
"""export checkpoint file into air models"""

import argparse
import numpy as np

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export

from src.config import config
from src.model import AttentionLstm

parser = argparse.ArgumentParser(description="ATAE_LSTM export")
parser.add_argument("--existed_ckpt",
                    type=str,
                    default="train/atae-lstm_max.ckpt",
                    help="existed checkpoint address.")
parser.add_argument("--file_name",
                    type=str,
                    default="ATAE_LSTM",
                    help="output file name.")
parser.add_argument("--file_format",
                    type=str,
                    choices=["AIR", "ONNX", "MINDIR"],
                    default="MINDIR",
                    help="file format")
args = parser.parse_args()

context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target="Ascend",
    device_id=6
)

if __name__ == "__main__":
    existed_ckpt = args.existed_ckpt

    r = np.load(config.word_vector)
    word_vector = r['weight']
    weight = Tensor(word_vector, mstype.float32)

    network = AttentionLstm(config, weight, is_train=False)

    model_path = existed_ckpt
    ms_ckpt = load_checkpoint(model_path)
    load_param_into_net(network, ms_ckpt)

    x = Tensor(np.zeros((1, 50)).astype(np.int32))
    x_len = Tensor(np.array([20]).astype(np.int32))
    aspect = Tensor(np.array([1]).astype(np.int32))

    export(network, x, x_len, aspect, file_name=args.file_name, file_format=args.file_format)
