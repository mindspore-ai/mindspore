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

import os
import time
from easydict import EasyDict
import numpy as np
from mindspore_serving.server import register
from mindspore_serving.server import distributed

from pangu.tokenization_jieba import JIEBATokenizer

cur_dir = os.path.abspath(os.path.dirname(__file__))
tokenizer_path = os.path.join(cur_dir, "tokenizer")
tokenizer = JIEBATokenizer(os.path.join(tokenizer_path, "vocab.vocab"), os.path.join(tokenizer_path, "vocab.model"))
end_token = tokenizer.eot_id

config = EasyDict({
    'frequency_penalty': 1.5,
    'presence_penalty': 0.3,
    'max_generate_length': 500,
    'top_k_num': 3,
    'top_p': 1.0,
    'end_token': 9,
    'seq_length': 1024,
    'vocab_size': 40000,
})


def topk_fun(logits, topk=5):
    """Get topk"""
    target_column = logits[0].tolist()
    sorted_array = [(k, v) for k, v in enumerate(target_column)]
    sorted_array.sort(key=lambda x: x[1], reverse=True)
    topk_array = sorted_array[:topk]
    index, value = zip(*topk_array)
    index = np.array([index])
    value = np.array([value])
    return value, index


distributed.declare_servable(rank_size=8, stage_size=1, with_batch_dim=False)


@register.register_method(output_names=["logits"])
def predict_sub0(input_ids, current_index, init, batch_valid_length):
    logits = register.call_servable(input_ids, current_index, init, batch_valid_length, subgraph=0)
    return logits


@register.register_method(output_names=["logits"])
def predict_sub1(input_id, current_index, init, batch_valid_length):
    logits = register.call_servable(input_id, current_index, init, batch_valid_length, subgraph=1)
    return logits


sub0_servable = register.PipelineServable(servable_name="pangu", method="predict_sub0")
sub1_servable = register.PipelineServable(servable_name="pangu", method="predict_sub1")


@register.register_pipeline(output_names=["output_sentence"])
def predict(input_sentence):
    """generate sentence with given input_sentence"""

    print(f"----------------------------- begin {input_sentence} ---------", flush=True)
    time_start = time.time()

    tokens = tokenizer.tokenize(input_sentence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    outputs = generate_increment(input_ids)

    return_tokens = tokenizer.convert_ids_to_tokens(outputs)
    reply = "".join(return_tokens)

    print(f"time cost {(time.time() - time_start) * 1000}ms, request '{input_sentence}' get reply '{reply}'",
          flush=True)

    return reply


def generate_increment(origin_inputs):
    """
    Text generation for incremental inference

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
    vocab_size = config.vocab_size

    # Init outputs with original inputs
    outputs = origin_inputs
    origin_inputs = np.array([origin_inputs])
    _, valid_length = origin_inputs.shape
    # If target length exceeds seq_length, use seq_length instead
    target_length = valid_length + max_generate_length
    target_length = seq_length if target_length > seq_length else target_length

    # A list of the frequency of each token
    frequency_list = np.array([[0 for _ in range(vocab_size)]])
    pad_length = seq_length - origin_inputs.shape[-1]
    # Pad original inputs to seq_length
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))

    # Indicate the exact token position
    current_index = valid_length - 1 if valid_length - 1 > 0 else 0
    current_index = np.array([current_index], np.int32)
    batch_valid_length = np.array([current_index], np.int32)
    # For first graph, not_init should be false
    init_true = True
    init_false = False
    init = init_false
    # Call a single inference with input size of (bs, seq_length)
    logits = sub0_servable.run(np.array(input_ids, np.int32), current_index, init, batch_valid_length)

    # Claim the second graph and set not_init to true
    init = init_true

    # A single loop generates one token, loop until reaching target seq_length or generating eod token
    while valid_length < target_length:
        # Reshape the output logits
        log_probs = logits.reshape(1, vocab_size)

        # Get the revised log_probs considering frequency and presence penalty to eliminate duplicate in generated results
        log_probs = log_probs.reshape(1, vocab_size)
        log_probs_revised = log_probs - frequency_list * frequency_penalty - (frequency_list > 0) * presence_penalty

        # Convert the log_probs to probability
        logits = np.power(10, np.array(log_probs_revised, np.float32))

        # If top_p is less than 1.0, use top_p sampling
        if top_p < 1.0:
            # Only consider the 5000 largest logits to reduce computation
            sorted_logits, index = topk_fun(logits, 5000)
            cumsum_logits = np.cumsum(sorted_logits, 1)
            cumsum_logits = cumsum_logits[0]
            index = index[0]
            sorted_logits = sorted_logits[0]
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
            probs, p_args = topk_fun(logits, top_k_num)
            probs = probs[0]
            p_args = p_args[0]
            # Avoid rounding error
            if sum(probs) == 0:
                probs = np.array([1 / top_k_num for _ in range(top_k_num)])
            p = probs / sum(probs)

        # Random select a token as final output for this round
        target_index = np.random.choice(len(p), p=p)
        # Stop judgment
        if p_args[target_index] == end_token or valid_length == target_length - 1:
            break

        # Update frequency list
        target = p_args[target_index]
        frequency_list[0][target] = frequency_list[0][target] + 1
        valid_length += 1

        batch_valid_length = np.array([valid_length - 1], np.int32)
        current_index = np.array([0], np.int32)
        input_id = np.array([[target]], np.int32)
        # Update outputs with current generated token
        outputs.append(int(target))

        # Call a single inference with input size of (bs, 1)
        logits = sub1_servable.run(input_id, current_index, init, batch_valid_length)
    # Return valid outputs out of padded outputs
    return outputs
