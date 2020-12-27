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
"""export checkpoint file into air models"""

import argparse
import numpy as np

from mindspore import Tensor, context, Parameter
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export

from config import GNMTConfig
from src.gnmt_model.gnmt import GNMT
from src.gnmt_model.gnmt_for_infer import GNMTInferCell
from src.utils import zero_weight
from src.utils.load_weights import load_infer_weights

parser = argparse.ArgumentParser(description="gnmt_v2 export")
parser.add_argument("--file_name", type=str, default="gnmt_v2", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument('--infer_config', type=str, required=True, help='gnmt_v2 config file')
parser.add_argument("--existed_ckpt", type=str, required=True, help="existed checkpoint address.")
parser.add_argument('--vocab_file', type=str, required=True, help='vocabulary file')
parser.add_argument("--bpe_codes", type=str, required=True, help="bpe codes to use.")
args = parser.parse_args()

context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target="Ascend",
    reserve_class_name_in_scope=False)


def get_config(config_file):
    tfm_config = GNMTConfig.from_json_file(config_file)
    tfm_config.compute_type = mstype.float16
    tfm_config.dtype = mstype.float32
    return tfm_config


if __name__ == '__main__':
    config = get_config(args.infer_config)
    config.existed_ckpt = args.existed_ckpt
    vocab = args.vocab_file
    bpe_codes = args.bpe_codes

    tfm_model = GNMT(config=config,
                     is_training=False,
                     use_one_hot_embeddings=False)

    params = tfm_model.trainable_params()
    weights = load_infer_weights(config)

    for param in params:
        value = param.data
        weights_name = param.name
        if weights_name not in weights:
            raise ValueError(f"{weights_name} is not found in weights.")
        if isinstance(value, Tensor):
            if weights_name in weights:
                assert weights_name in weights
                if isinstance(weights[weights_name], Parameter):
                    if param.data.dtype == "Float32":
                        param.set_data(Tensor(weights[weights_name].data.asnumpy(), mstype.float32))
                    elif param.data.dtype == "Float16":
                        param.set_data(Tensor(weights[weights_name].data.asnumpy(), mstype.float16))

                elif isinstance(weights[weights_name], Tensor):
                    param.set_data(Tensor(weights[weights_name].asnumpy(), config.dtype))
                elif isinstance(weights[weights_name], np.ndarray):
                    param.set_data(Tensor(weights[weights_name], config.dtype))
                else:
                    param.set_data(weights[weights_name])
            else:
                print("weight not found in checkpoint: " + weights_name)
                param.set_data(zero_weight(value.asnumpy().shape))

    print(" | Load weights successfully.")
    tfm_infer = GNMTInferCell(tfm_model)
    tfm_infer.set_train(False)

    source_ids = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))
    source_mask = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))

    export(tfm_infer, source_ids, source_mask, file_name=args.file_name, file_format=args.file_format)
