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
import numpy as np

from mindspore.common import dtype as mstype

from config import TransformerConfig
from src.transformer import infer
from src.utils import ngram_ppl
from src.utils import Dictionary
from src.utils import rouge

parser = argparse.ArgumentParser(description='Evaluation MASS.')
parser.add_argument("--config", type=str, required=True,
                    help="Model config json file path.")
parser.add_argument("--vocab", type=str, required=True,
                    help="Vocabulary to use.")
parser.add_argument("--output", type=str, required=True,
                    help="Result file path.")


def get_config(config):
    config = TransformerConfig.from_json_file(config)
    config.compute_type = mstype.float16
    config.dtype = mstype.float32
    return config


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    vocab = Dictionary.load_from_persisted_dict(args.vocab)
    _config = get_config(args.config)
    result = infer(_config)
    with open(args.output, "wb") as f:
        pickle.dump(result, f, 1)

    ppl_score = 0.
    preds = []
    tgts = []
    _count = 0
    for sample in result:
        sentence_prob = np.array(sample['prediction_prob'], dtype=np.float32)
        sentence_prob = sentence_prob[:, 1:]
        _ppl = []
        for path in sentence_prob:
            _ppl.append(ngram_ppl(path, log_softmax=True))
        ppl = np.min(_ppl)
        preds.append(' '.join([vocab[t] for t in sample['prediction']]))
        tgts.append(' '.join([vocab[t] for t in sample['target']]))
        print(f" | source: {' '.join([vocab[t] for t in sample['source']])}")
        print(f" | target: {tgts[-1]}")
        print(f" | prediction: {preds[-1]}")
        print(f" | ppl: {ppl}.")
        if np.isinf(ppl):
            continue
        ppl_score += ppl
        _count += 1

    print(f" | PPL={ppl_score / _count}.")
    rouge(preds, tgts)
