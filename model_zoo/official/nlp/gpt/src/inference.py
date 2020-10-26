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

"""
TopK for text generation
"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

def generate(model, origin_inputs, seq_length, end_token=50256):
    """
    TopK for text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        seq_length: seq_length for the model
        end_token: end of sentence token id

    Returns:
        outputs: the ids for the generated text
    """
    TOPK = 5
    seq_length = seq_length
    bs, valid_length = origin_inputs.shape
    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))
    print("input_ids is ", input_ids)
    while valid_length < seq_length:
        inputs = Tensor(input_ids, mstype.int32)
        logits = model(inputs).asnumpy()
        logits = logits.reshape(bs, seq_length, -1)
        probs = logits[0, valid_length-1, :]
        p_args = probs.argsort()[::-1][:TOPK]

        p = probs[p_args]
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        if p_args[target_index] == end_token or valid_length == seq_length-1:
            outputs = input_ids
            break
        input_ids[0][valid_length] = p_args[target_index]
        valid_length += 1
    length = np.sum(outputs != 0)
    outputs = outputs[0][:length]
    return outputs
