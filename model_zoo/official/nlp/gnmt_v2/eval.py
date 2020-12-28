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
"""Evaluation api."""
import argparse
import pickle
import os

from mindspore.common import dtype as mstype

from config import GNMTConfig
from src.gnmt_model import infer
from src.gnmt_model.bleu_calculate import bleu_calculate
from src.dataset.tokenizer import Tokenizer

parser = argparse.ArgumentParser(description='gnmt')
parser.add_argument("--config", type=str, required=True,
                    help="model config json file path.")
parser.add_argument("--test_dataset", type=str, required=True,
                    help="test dataset address.")
parser.add_argument("--existed_ckpt", type=str, required=True,
                    help="existed checkpoint address.")
parser.add_argument("--vocab", type=str, required=True,
                    help="Vocabulary to use.")
parser.add_argument("--bpe_codes", type=str, required=True,
                    help="bpe codes to use.")
parser.add_argument("--test_tgt", type=str, required=True,
                    default=None,
                    help="data file of the test target")
parser.add_argument("--output", type=str, required=False,
                    default="./output.npz",
                    help="result file path.")


def get_config(config):
    config = GNMTConfig.from_json_file(config)
    config.compute_type = mstype.float16
    config.dtype = mstype.float32
    return config


def _check_args(config):
    if not os.path.exists(config):
        raise FileNotFoundError("`config` is not existed.")
    if not isinstance(config, str):
        raise ValueError("`config` must be type of str.")


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    _check_args(args.config)
    _config = get_config(args.config)
    _config.test_dataset = args.test_dataset
    _config.existed_ckpt = args.existed_ckpt
    result = infer(_config)

    with open(args.output, "wb") as f:
        pickle.dump(result, f, 1)

    result_npy_addr = args.output
    vocab = args.vocab
    bpe_codes = args.bpe_codes
    test_tgt = args.test_tgt
    tokenizer = Tokenizer(vocab, bpe_codes, 'en', 'de')
    scores = bleu_calculate(tokenizer, result_npy_addr, test_tgt)
    print(f"BLEU scores is :{scores}")
