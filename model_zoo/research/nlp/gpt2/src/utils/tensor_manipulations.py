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
tensor manipulations
"""
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P


def extract_string_from_tensor(input_ids, mode="single", config=None, tokenizer=None):
    """
    Args:
        input_ids (Tensor): input sentences with shape [batch_size, seq_len].
        mode (str): ["pair", "single"]
                    "pair" for tasks with paired inputs `<bos> A <eos> B <eos>`,
                    such as summarization task, the dataset format `<bos> Article <eos> Summary <eos>`,
                    reading comprehension task, the dataset format `<bos> Passage Question <eos> Answer <eos>`.

                    "single" for tasks with single input `<bos> A <eos>`, such as Language Modeling, Lambada task.
        config: the configuration of GPT-2 model.
        tokenizer: the tokenizer of GPT-2 model.

    Return:
        prompt_list (list): list of prompt_text
        reference_list (list): list of reference_text, or second part of text
        rest_list (list): list of rest_text, or rest part of text

    """

    batch_size = config.batch_size
    seq_length = config.seq_length
    prompt_list = [""] * batch_size
    reference_list = [""] * batch_size
    eos_text = tokenizer.eos_token
    len_eos_text = len(eos_text)
    input_ids = P.Reshape()(input_ids, (batch_size, seq_length))

    if mode == "pair":

        for batch_idx in range(batch_size):
            sentence_tensor = input_ids[batch_idx]
            sentence_list = sentence_tensor.asnumpy().tolist()[1:]

            sentence = tokenizer.decode(sentence_list)
            prompt_start = 0
            prompt_end = sentence.find(eos_text, 0)
            reference_start = prompt_end + len_eos_text
            reference_end = sentence[reference_start:].find(
                eos_text, 0) + reference_start
            prompt_list[batch_idx] = sentence[prompt_start:prompt_end]
            reference_list[batch_idx] = sentence[reference_start:reference_end]

        return prompt_list, reference_list

    # For single output datasets such as WikiText, etc.
    if mode == "single":
        for batch_idx in range(batch_size):
            sentence_tensor = input_ids[batch_idx]
            sentence_list = sentence_tensor.asnumpy().tolist()[1:]

            sentence = tokenizer.decode(sentence_list)
            prompt_start = 0
            prompt_end = sentence.find(eos_text, 0)
            prompt_list[batch_idx] = sentence[prompt_start:prompt_end]
    else:
        raise NotImplementedError('mode:{} not supported.'.format(mode))

    return prompt_list


def extract_single_token_logits(logits=None, seq_pos=None):
    """
    Args
        logits: (batch_size,seq_length,vocab_size) e.g. when batchsize is 8,
        sequence length is 1024 and vocab_size is 50257,
        then logits is a Tensor with shape (8,1024,50257)
        seq_pos:(batch_size) list
    Return:
        output_logits: (batch_size,1,vocab_size) extract the logit to predict the last token.
    """

    batch_size = logits.shape[0]
    for i in range(batch_size):
        logit = logits[i:i + 1:1, seq_pos[i]:seq_pos[i] + 1:1, ::]
        if i == 0:
            output_logits = logit
        else:
            output_logits = P.Concat()((output_logits, logit))

    return output_logits


def get_last_one_pos(input_mask: Tensor):
    """
    Arg:
        input_mask (Tensor): (batch_size,seq_length)
    Return:
        pos (Tensor): (batch_size,)
    """
    input_mask_ = P.Cast()(input_mask, mstype.float32)
    pos = P.ReduceSum(keep_dims=False)(input_mask_, axis=1)  # (batch_size,)
    pos = P.Cast()(pos, mstype.int32)
    pos = pos - 1
    return pos


def get_next_one_pos(input_mask: Tensor):
    """
    Arg:
        input_mask (Tensor): (batch_size,seq_length)
    """
    input_mask_ = P.Cast()(input_mask, mstype.float32)
    pos = P.ReduceSum(keep_dims=False)(input_mask_, axis=1)  # (batch_size,)
    pos = P.Cast()(pos, mstype.int32)
    return pos


def add_last_token_mask(input_mask: Tensor, overflow_strategy: str = "shift"):
    """
    add last token mask
    Args:
        input_mask: Tensor
        overflow_strategy: str

    Returns:
        Tensor

    """
    pos = get_next_one_pos(input_mask).asnumpy()
    input_mask_np = input_mask.asnumpy()
    maximum_length = input_mask.shape[1]
    batch_size = input_mask.shape[0]
    for idx in range(batch_size):
        # not overflow
        if pos[idx] < maximum_length:
            input_mask_np[idx][pos[idx]] = 1

        # overflow
        else:
            if overflow_strategy == "shift":
                continue
            if overflow_strategy == "truncate":
                continue
            else:
                raise ValueError("{} is not an option in ['shift','truncate'].".format(overflow_strategy))
    return Tensor(input_mask_np, dtype=mstype.int32)


def add_last_token(input_ids: Tensor, input_mask: Tensor, overflow_strategy: str = "shift", append_ids=None,
                   next_token_pos=None):
    """
    add last token
    Args:
        input_ids: Tensor
        input_mask: Tensor
        overflow_strategy: str
        append_ids: Any
        next_token_pos: Any

    Returns:
        Tensor

    """
    # get positional list/numpy array
    if next_token_pos is None:
        pos = get_next_one_pos(input_mask).asnumpy()
    else:
        pos = next_token_pos
    # get numpy of inputs
    input_mask_np = input_mask.asnumpy()
    input_ids_np = input_ids.asnumpy()
    maximum_length = int(input_mask.shape[1])
    batch_size = int(input_mask.shape[0])

    for idx in range(batch_size):
        if append_ids[idx] == -1:
            continue
        # not overflow
        if pos[idx] < maximum_length:
            input_mask_np[idx][int(pos[idx])] = 1
            input_ids_np[idx][int(pos[idx])] = append_ids[idx]

        # overflow
        else:
            if overflow_strategy == "shift":
                # shift one token left
                input_ids_np[idx][0:maximum_length - 1] = input_ids_np[idx][1:maximum_length]
                input_ids_np[idx][maximum_length - 1] = append_ids[idx]
                continue
            if overflow_strategy == "truncate":
                # do nothing
                continue
            else:
                raise ValueError("{} is not an option in ['shift','truncate'].".format(overflow_strategy))
    return Tensor(input_ids_np, dtype=mstype.int32), Tensor(input_mask_np, dtype=mstype.int32)
