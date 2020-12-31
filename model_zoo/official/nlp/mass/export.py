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

from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export

from src.utils import Dictionary
from src.utils.load_weights import load_infer_weights
from src.transformer.transformer_for_infer import TransformerInferModel
from config import TransformerConfig

parser = argparse.ArgumentParser(description="mass export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--file_name", type=str, default="mass", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
parser.add_argument('--gigaword_infer_config', type=str, required=True, help='gigaword config file')
parser.add_argument('--vocab_file', type=str, required=True, help='vocabulary file')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

def get_config(config_file):
    tfm_config = TransformerConfig.from_json_file(config_file)
    tfm_config.compute_type = mstype.float16
    tfm_config.dtype = mstype.float32

    return tfm_config

if __name__ == '__main__':
    vocab = Dictionary.load_from_persisted_dict(args.vocab_file)
    config = get_config(args.gigaword_infer_config)
    dec_len = config.max_decode_length

    tfm_model = TransformerInferModel(config=config, use_one_hot_embeddings=False)
    tfm_model.init_parameters_data()

    params = tfm_model.trainable_params()
    weights = load_infer_weights(config)

    for param in params:
        value = param.data
        name = param.name

        if name not in weights:
            raise ValueError(f'{name} is not found in weights.')

        with open('weight_after_deal.txt', 'a+') as f:
            weights_name = name
            f.write(weights_name + '\n')

            if isinstance(value, Tensor):
                if weights_name in weights:
                    assert weights_name in weights
                    param.set_data(Tensor(weights[weights_name], mstype.float32))
                else:
                    raise ValueError(f'{weights_name} is not found in checkpoint')
            else:
                raise TypeError(f'Type of {weights_name} is not Tensor')

    print('    |    Load weights successfully.')
    tfm_model.set_train(False)

    source_ids = Tensor(np.ones((1, config.seq_length)).astype(np.int32))
    source_mask = Tensor(np.ones((1, config.seq_length)).astype(np.int32))

    export(tfm_model, source_ids, source_mask, file_name=args.file_name, file_format=args.file_format)
