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
"""servable config for pangu alpha"""


from mindspore_serving.worker import register
from mindspore_serving.worker import distributed
import numpy as np

# define preprocess pipeline, the function arg is multi instances, every instance is tuple of inputs
# this example has one input and one output

seq_length = 1024


def preprocess(input_tokens):
    """Preprocess, padding for input"""
    _, valid_length = input_tokens.shape
    token_ids = np.pad(input_tokens, ((0, 0), (0, seq_length - valid_length)), 'constant', constant_values=(0, 6))
    token_ids = token_ids.astype(np.int32)
    return token_ids, valid_length


def topk_fun(logits, valid_length, topk=5):
    """Get topk"""
    target_column = logits[valid_length - 1, :].tolist()
    sorted_array = [(k, v) for k, v in enumerate(target_column)]
    sorted_array.sort(key=lambda x: x[1], reverse=True)
    topk_array = sorted_array[:topk]
    index, value = zip(*topk_array)
    index = np.array(index)
    value = np.array(value)
    return index, value


def postprocess_topk(logits, valid_length):
    """Postprocess for one output"""
    p_args, p = topk_fun(logits, valid_length, 5)
    p = p / sum(p)
    target_index = np.random.choice(len(p), p=p)
    target = p_args[target_index]
    return target


def postprocess(p, p_args, valid_length):
    """Postprocess for two output"""
    p = p[valid_length - 1]
    p_args = p_args[valid_length - 1]
    p = p / sum(p)
    target_index = np.random.choice(len(p), p=p)
    target = p_args[target_index]
    return target


distributed.declare_distributed_servable(rank_size=8, stage_size=1, with_batch_dim=False)


@register.register_method(output_names=["add_token"])
def predict(input_tokens):
    """register predict method in pangu-alpha"""
    token_ids, valid_length = register.call_preprocess(preprocess, input_tokens)
    ############# two output ###################
    # p, p_args = register.call_servable(token_ids)
    # add_token = register.call_postprocess(postprocess, p, p_args, valid_length)
    #############################################

    ################# one output ####################
    logits = register.call_servable(token_ids)
    add_token = register.call_postprocess(postprocess_topk, logits, valid_length)
    return add_token
