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

"""
TopK for text generation
"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P

def generate(model, origin_inputs, config):
    """
    TopK for text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        config: inference configurations

    Returns:
        outputs: the ids for the generated text
    """
    # Get configurations for inference
    frequency_penalty = config.frequency_penalty
    presence_penalty = config.presence_penalty
    top_p = config.top_p
    top_k_num = config.top_k_num
    max_generate_length = config.max_generate_length
    seq_length = config.seq_length
    end_token = config.end_token

    _, valid_length = origin_inputs.shape
    # If target length exceeds seq_length, use seq_length instead
    target_length = valid_length + max_generate_length
    target_length = seq_length if target_length > seq_length else target_length

    # A list of the frequency of each token
    frequency_list = np.array([[0 for _ in range(config.vocab_size)]])
    pad_length = seq_length - origin_inputs.shape[-1]
    # Pad original inputs to seq_length
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))
    print("input_ids is ", input_ids)

    # A single loop generates one token, loop until reaching target seq_length or generating eod token
    while valid_length < target_length:
        inputs = Tensor(input_ids, mstype.int32)
        # Indicate the exact token position
        current_index = valid_length - 1 if valid_length - 1 > 0 else 0
        current_index = Tensor([current_index], mstype.int32)
        # Call a single inference
        log_probs = model.predict(inputs, current_index)
        # Get the revised log_probs considering frequency and presence penalty to eliminate duplicate in generated results
        log_probs = log_probs.asnumpy().reshape(1, config.vocab_size)
        log_probs_revised = log_probs - frequency_list * frequency_penalty - (frequency_list > 0) * presence_penalty

        # Convert the log_probs to probability
        logits = P.Pow()(10, Tensor(log_probs_revised, mstype.float32))

        # If top_p is less than 1.0, use top_p sampling
        if top_p < 1.0:
            # Only consider the 5000 largest logits to reduce computation
            sorted_logits, index = P.TopK(sorted=True)(logits, 5000)
            cumsum_logits = P.CumSum()(sorted_logits, 1)
            cumsum_logits = cumsum_logits.asnumpy()[0]
            index = index.asnumpy()[0]
            sorted_logits = sorted_logits.asnumpy()[0]
            top_p_num = sum(cumsum_logits > top_p)
            # In case the probability is smooth, the sum of 5000 largest probabilities are not large enough
            if top_p_num == 0:
                top_p_num = 5000
            # Get the corresponding probs and indices
            probs = sorted_logits[:top_p_num]
            p_args = index[:top_p_num]
            p = probs / sum(probs)
        # if top_p is set to 1.0, use top_k sampling
        else:
            # Get the corresponding probs and indices
            probs, p_args = P.TopK(sorted=True)(logits, top_k_num)
            probs = probs.asnumpy()[0]
            p_args = p_args.asnumpy()[0]
            # Avoid rounding error
            if sum(probs) == 0:
                probs = np.array([1 / top_k_num for _ in range(top_k_num)])
            p = probs / sum(probs)

        # Random select a token as final output for this round
        target_index = np.random.choice(len(p), p=p)
        # Stop judgment
        if p_args[target_index] == end_token or valid_length == target_length-1:
            outputs = input_ids
            break
        # Modify input_ids with newly generated token
        input_ids[0][valid_length] = p_args[target_index]
        valid_length += 1
    # Return valid outputs out of padded outputs
    length = np.sum(outputs != 0)
    outputs = outputs[0][:length]
    return outputs
