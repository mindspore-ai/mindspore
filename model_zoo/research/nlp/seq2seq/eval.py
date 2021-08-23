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
"""Evaluation api."""
import os
# os.system("pip3 install subword-nmt")
# os.system("pip3 install sacremoses")
import ast
import argparse
import pickle
from mindspore.common import dtype as mstype
from mindspore import context

from config import Seq2seqConfig
from src.seq2seq_model import infer
from src.seq2seq_model.bleu_calculate import bleu_calculate
from src.dataset.tokenizer import Tokenizer

is_modelarts = False


parser = argparse.ArgumentParser(description='seq2seq')
parser.add_argument("--config", type=str, required=True,
                    help="model config json file path.")
parser.add_argument("--data_url", type=str, default=None,
                    help="data address.")
parser.add_argument("--train_url", type=str, default=None,
                    help="output address.")
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
parser.add_argument("--is_modelarts", type=ast.literal_eval, default=False,
                    help="running on modelarts")
args, _ = parser.parse_known_args()
if args.is_modelarts:
    import moxing as mox

context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=True,
    device_target="Ascend",
    reserve_class_name_in_scope=True)

def get_config(config):
    config = Seq2seqConfig.from_json_file(config)
    config.compute_type = mstype.float16
    config.dtype = mstype.float32
    return config

def _check_args(config):
    if not os.path.exists(config):
        raise FileNotFoundError("`config` is not existed.")
    if not isinstance(config, str):
        raise ValueError("`config` must be type of str.")


if __name__ == '__main__':
    _check_args(args.config)
    _config = get_config(args.config)

    if args.is_modelarts:
        mox.file.copy_parallel(src_url=args.data_url, dst_url='/cache/dataset_menu/')
        _config.test_dataset = '/cache/dataset_menu/newstest2014.en.mindrecord'
        _config.existed_ckpt = '/cache/dataset_menu/seq2seq-7_1642.ckpt'

    _config.test_dataset = args.test_dataset
    _config.existed_ckpt = args.existed_ckpt

    result = infer(_config)

    with open(args.output, "wb") as f:
        pickle.dump(result, f, 1)

    result_npy_addr = args.output
    vocab = args.vocab
    bpe_codes = args.bpe_codes
    test_tgt = args.test_tgt
    tokenizer = Tokenizer(vocab, bpe_codes, 'en', 'fr')
    scores = bleu_calculate(tokenizer, result_npy_addr, test_tgt)
    print(f"BLEU scores is :{scores}")

    if args.is_modelarts:
        result_npy_addr = output
        vocab = '/cache/dataset_menu/vocab.bpe.32000'
        bpe_codes = '/cache/dataset_menu/bpe.32000'
        test_tgt = '/cache/dataset_menu/newstest2014.fr'
        tokenizer = Tokenizer(vocab, bpe_codes, 'en', 'fr')
        scores = bleu_calculate(tokenizer, result_npy_addr, test_tgt)
        print(f"BLEU scores is :{scores}")
        mox.file.copy_parallel(src_url='/cache/infer_output/', dst_url=args.train_url)
        