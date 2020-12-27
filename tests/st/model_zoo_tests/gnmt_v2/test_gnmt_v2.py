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
"""Train and eval api."""
import os
import argparse
import pickle
import datetime

import mindspore.common.dtype as mstype
from mindspore.common import set_seed

from config import GNMTConfig
from train import train_parallel
from src.gnmt_model import infer
from src.gnmt_model.bleu_calculate import bleu_calculate
from src.dataset.tokenizer import Tokenizer

parser = argparse.ArgumentParser(description='GNMT train and eval.')
# train
parser.add_argument("--config_train", type=str, required=True,
                    help="model config json file path.")
parser.add_argument("--pre_train_dataset", type=str, required=True,
                    help="pre-train dataset address.")
# eval
parser.add_argument("--config_test", type=str, required=True,
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
    start_time = datetime.datetime.now()
    _rank_size = os.getenv('RANK_SIZE')
    args, _ = parser.parse_known_args()
    # train
    _check_args(args.config_train)
    _config_train = get_config(args.config_train)
    _config_train.pre_train_dataset = args.pre_train_dataset
    set_seed(_config_train.random_seed)
    assert _rank_size is not None and int(_rank_size) > 1
    if _rank_size is not None and int(_rank_size) > 1:
        train_parallel(_config_train)
    # eval
    _check_args(args.config_test)
    _config_test = get_config(args.config_test)
    _config_test.test_dataset = args.test_dataset
    _config_test.existed_ckpt = args.existed_ckpt
    result = infer(_config_test)

    with open(args.output, "wb") as f:
        pickle.dump(result, f, 1)

    result_npy_addr = args.output
    vocab = args.vocab
    bpe_codes = args.bpe_codes
    test_tgt = args.test_tgt
    tokenizer = Tokenizer(vocab, bpe_codes, 'en', 'de')
    scores = bleu_calculate(tokenizer, result_npy_addr, test_tgt)
    print(f"BLEU scores is :{scores}")
    end_time = datetime.datetime.now()
    cost_time = (end_time - start_time).seconds
    print(f"Cost time is {cost_time}s.")
    assert scores >= 23.8
    assert cost_time < 10800.0
    print("----done!----")
