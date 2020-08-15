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
import os
import argparse
import pickle

from mindspore.common import dtype as mstype
from mindspore import context

from config import TransformerConfig
from src.transformer import infer, infer_ppl
from src.utils import Dictionary
from src.utils import get_score

parser = argparse.ArgumentParser(description='Evaluation MASS.')
parser.add_argument("--config", type=str, required=True,
                    help="Model config json file path.")
parser.add_argument("--vocab", type=str, required=True,
                    help="Vocabulary to use.")
parser.add_argument("--output", type=str, required=True,
                    help="Result file path.")
parser.add_argument("--metric", type=str, default='rouge',
                    help='Set eval method.')
parser.add_argument("--platform", type=str, required=True,
                    help="model working platform.")


def get_config(config):
    config = TransformerConfig.from_json_file(config)
    config.compute_type = mstype.float16
    config.dtype = mstype.float32
    return config


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    vocab = Dictionary.load_from_persisted_dict(args.vocab)
    _config = get_config(args.config)

    device_id = os.getenv('DEVICE_ID', None)
    if device_id is None:
        device_id = 0
    device_id = int(device_id)
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args.platform,
        reserve_class_name_in_scope=False,
        device_id=device_id)

    if args.metric == 'rouge':
        result = infer(_config)
    else:
        result = infer_ppl(_config)

    with open(args.output, "wb") as f:
        pickle.dump(result, f, 1)

    # get score by given metric
    score = get_score(result, vocab, metric=args.metric)
    print(score)
