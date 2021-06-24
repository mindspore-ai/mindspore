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
import pickle

from mindspore import context
from mindspore.common import dtype as mstype
from src.transformer import infer, infer_ppl
from src.utils import Dictionary
from src.utils import get_score
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

def get_config():
    if config.compute_type == "float16":
        config.compute_type = mstype.float16
    if config.compute_type == "float32":
        config.compute_type = mstype.float32
    if config.dtype == "float16":
        config.dtype = mstype.float16
    if config.dtype == "float32":
        config.dtype = mstype.float32

@moxing_wrapper()
def eval_net():
    """eval_net"""
    vocab = Dictionary.load_from_persisted_dict(config.vocab)
    get_config()
    device_id = os.getenv('DEVICE_ID', None)
    if device_id is None:
        device_id = 0
    device_id = int(device_id)
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.device_target,
        reserve_class_name_in_scope=False,
        device_id=device_id)

    if config.metric == 'rouge':
        result = infer(config)
    else:
        result = infer_ppl(config)

    with open(config.output, "wb") as f:
        pickle.dump(result, f, 1)

    # get score by given metric
    score = get_score(result, vocab, metric=config.metric)
    print(score)

if __name__ == '__main__':
    eval_net()
