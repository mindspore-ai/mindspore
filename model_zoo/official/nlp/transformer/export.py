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
""" export checkpoint file into models"""

import argparse
import numpy as np

from mindspore import Tensor, context
from mindspore.train.serialization import load_param_into_net, export

from src.transformer_model import TransformerModel
from src.eval_config import cfg, transformer_net_cfg
from eval import load_weights

parser = argparse.ArgumentParser(description='transformer export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--file_name", type=str, default="transformer", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', help='file format')
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    tfm_model = TransformerModel(config=transformer_net_cfg, is_training=False, use_one_hot_embeddings=False)

    parameter_dict = load_weights(cfg.model_file)
    load_param_into_net(tfm_model, parameter_dict)

    source_ids = Tensor(np.ones((transformer_net_cfg.batch_size, transformer_net_cfg.seq_length)).astype(np.int32))
    source_mask = Tensor(np.ones((transformer_net_cfg.batch_size, transformer_net_cfg.seq_length)).astype(np.int32))

    export(tfm_model, source_ids, source_mask, file_name=args.file_name, file_format=args.file_format)
