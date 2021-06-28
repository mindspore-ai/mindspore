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
task utils
"""
import regex as re

from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor


# for lambada task
def extract_logits(logits=None, position=None):
    """
    Args
        logits (Tensor): Tensor(batch_size,seq_length,vocab_size) e.g.(8,1024,50257)
        position (numpy.array): the array stored the fianl word position, shape with [batch_size, 2]

    Return:
        output_logits (Tensor): extract the Specified logit according to the position,
                                shape with [batch_size, vocab_size]
    """

    batch_size = logits.shape[0]

    for batch_idx in range(batch_size):
        word_logits_pos = int(position[batch_idx, 0] - 1)

        logit = logits[batch_idx:batch_idx+1:1, word_logits_pos, ::] # [1, vocab_size]
        if batch_idx == 0:
            output_logits = logit
        else:
            output_logits = P.Concat()((output_logits, logit)) # [batch_size, vocab_size]

    return output_logits


def get_final_word_label(input_ids, input_length, tokenizer=None):
    """
    get whole word label_str from input_ids
    Args:
        input_ids: Tensor(batch_size,seq_length), indices of input text
        config: GPT2Config, config of GPT2 model, if not initiated,
        this function will create a MockConfig by params of input_ids, optional
        tokenizer: GPT2Tokenizer, if not initiated, it will be created using the default setting in utils.tokenization,
        optional
    Returns:
        batch_word_label: [str], lastword str given lambada as label
    """
    input_ids_np = input_ids.asnumpy()
    input_length_np = input_length.asnumpy()
    batch_word_label = []

    for batch_idx in range(len(input_ids_np)):
        word_spos = input_length_np[batch_idx, 0]
        word_epos = input_length_np[batch_idx, 1]
        final_word_ids = input_ids_np[batch_idx, word_spos:word_epos]
        final_word_str = tokenizer.decode(final_word_ids.tolist())
        batch_word_label.append(final_word_str)

    return batch_word_label


def calculate_final_word_loss(logits, batch_size, input_ids, input_length, loss):
    """
    Calculate the last word loss.
    """
    logits = logits.asnumpy()
    input_len_np = input_length.asnumpy()
    input_ids_np = input_ids.asnumpy()

    sum_batch_loss = 0.0

    for batch in range(batch_size):
        lastword_spos = input_len_np[batch, 0]
        lastword_epos = input_len_np[batch, 1]

        last_word_logits = logits[batch, lastword_spos - 1:lastword_epos - 1:1, ::]
        last_word_logits_tensor = Tensor(last_word_logits, mstype.float32)

        last_word_label = input_ids_np[batch, lastword_spos:lastword_epos:1]
        print("last word label: ", last_word_label)
        last_word_label_tensor = Tensor(last_word_label, mstype.int32)

        last_word_loss = loss(last_word_logits_tensor, last_word_label_tensor)
        last_word_loss = float(last_word_loss.asnumpy())
        sum_batch_loss += last_word_loss
        print(" | loss: ", last_word_loss)

    avg_batch_loss = float(sum_batch_loss / batch_size)
    return avg_batch_loss


# for cbt task
def calculate_choice_prob_for_cbt(logits, batch_size, input_length, input_ids):
    """
    calculate choice prob for cbt
    Args:
        logits:
        batch_size: Any
        input_length: {asnumpy}
        input_ids: {asnumpy}

    Returns:
        choice_prob: List[float]

    """
    choice_prob = []  # [batch_size]
    logits = logits.asnumpy()
    input_len_np = input_length.asnumpy()
    input_ids_np = input_ids.asnumpy()

    for batch in range(batch_size):
        sum_ = 0.0
        rest_spos = input_len_np[batch, 0]
        rest_epos = input_len_np[batch, 1] + 1
        for rest_pos in range(rest_spos - 1, rest_epos - 1):
            rest_token_id = input_ids_np[batch, rest_pos + 1]
            log_prob = logits[batch, rest_pos, rest_token_id]
            sum_ = sum_ + log_prob
        choice_prob.append(sum_)
        print("rest sentence prob: ", sum_)

    return choice_prob


# for summarization task
def modify_paramdict(param_dict, mode="zero-shot", model_prefix="gpt2."):
    """
    modify keys of param_dict to fit model.

    Args:
        param_dic: dict, dictionary of parameters imported from a ckpt file
        mode:   str, "zero-shot" for an pretrained GPT2 model;
                "finetune" for an finetuned model for certain task.
    Return:
        reorganized_param_dict: dict, new param_dict to fit in model for different tasks.
    """
    final_param_dict = dict()
    if mode == "zero-shot":
        for name in param_dict:
            final_param_dict[model_prefix + name] = param_dict[name]
        final_param_dict['lm_head.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']
    elif mode == "finetuned":
        embedding_name = "gpt2_embedding_lookup.embedding_table"
        embedding_name_old = ""
        for name in param_dict:
            name_remove_prefix = name[len(model_prefix):]
            name_prefix = name[:len(model_prefix)]
            final_param_dict[name_remove_prefix] = param_dict[name]
            if embedding_name in name and name_prefix == model_prefix:
                embedding_name_old = name
        final_param_dict[embedding_name] = param_dict[embedding_name_old]
    else:
        raise ValueError("mode should be [zero-shot, finetuned]")
    return final_param_dict


def clean_hypo(text):
    """
    to prevent generation of empty string, and lower text

    Arg:
        text: str, input str
    Return:
        text: str, cleaned input str
    """
    text = text.lower()
    eng_re = re.compile(r'[a-z]+', re.I)
    length_con = len(eng_re.findall(text))
    if length_con == 0:
        return '<EMPTY>'
    return text
