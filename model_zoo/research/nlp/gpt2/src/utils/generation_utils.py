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
generation utils
"""

import numpy as np
from scipy.special import softmax

from mindspore.ops import operations as P
from mindspore import dtype as mstype
from mindspore.common.tensor import Tensor

from .tensor_manipulations import extract_single_token_logits, add_last_token

INF = 1. * 1e9


class TopKTopP_Filter():
    """
    Top K sampling along with Top P sampling(Nucleus Sampling)

    Choose top-K probability of ids and those with top-P probability ids into candidate sample sets.
    Use np.random.multinomial to sample

    Args:
        batch_size (int): batch size of input dataset.
        vocab_size (int): the shape of each embedding vector.
        k (int): parameter for Top-K sampling, k should be in range of [0, vocab_size].
                 0 for no filter for TopK sampling(do nothing). Default: 0.
        p (float) [Optional]: parameter for Top-P sampling a.k.a. Necleus Sampling, p is in between 0.0 and 1.0.
                   Default: 1.0.
        temperature (float) [Optional]: parameter for generation, greater if generation more diverse. Default: 1.0.

    """

    def __init__(self,
                 batch_size=None,
                 vocab_size=None,
                 k=0,
                 p=1.0,
                 temperature=1.0,
                 min_tokens_to_keep=1,
                 ):

        self.k = k
        self.p = p
        self.temp = temperature

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.min_tokens_to_keep = min_tokens_to_keep

        assert self.temp > 0.0, 'temperature must be positive'
        assert self.k >= 0, 'the top_k number must be no negative.'
        if self.k > 0:
            assert self.min_tokens_to_keep <= self.k, 'k must be larger than or equal to min_token_to_keep ' \
                                                      'for Top-p sampling'

        if self.k == 0:
            self.k = self.vocab_size

        self.safety_mask = np.concatenate((np.ones((self.batch_size, self.min_tokens_to_keep)),
                                           np.zeros((self.batch_size, self.k - self.min_tokens_to_keep))),
                                          axis=1).astype(np.bool)

    def calculate(self, distribution):
        """
        calculate sampling procedure with setting initialized before, return a list of sampled ids.

        Args:
            distribution (numpy.ndarray): with shape (batch_size,vocab_size)

        Returns:
            sampled ids: a list, with length of batch_size
        """

        if self.temp != 1.0:
            distribution = distribution / float(self.temp)

        distribution_sorted = -np.sort(-distribution, axis=1)
        index_sorted = np.argsort(-distribution, axis=1)

        topk_distribution = distribution_sorted[::, :self.k if self.k > 0 else self.vocab_size]
        topk_indices = index_sorted[::, :self.k if self.k > 0 else self.vocab_size]

        # safety check of probability
        self.p = max(0.0, min(1.0, self.p))
        cum_sum = np.cumsum(softmax(topk_distribution, axis=1), axis=1)
        bool_map = np.logical_or((cum_sum <= self.p), self.safety_mask).astype(np.float32)

        topk_distribution = topk_distribution * bool_map + np.float32(-1e5) * (1.0 - bool_map)
        topk_distribution = softmax(topk_distribution, axis=1)

        # normalize for np.float64
        # choose np.float64 to avoid overflow in softmax operation
        topk_distribution = topk_distribution.astype(np.float64)
        for batch_idx in range(self.batch_size):
            topk_distribution[batch_idx] = topk_distribution[batch_idx] / np.sum(topk_distribution[batch_idx])

        ret_ids = []
        for batch_idx in range(self.batch_size):
            select_index = np.argmax(np.random.multinomial(1, topk_distribution[batch_idx]))
            ret_ids.append(topk_indices[batch_idx][select_index])

        return ret_ids


class Sample():
    """
    Initiate a Sample object for sampling next token(s) from previous text.

    Args:
        decoder (Model): GPT2 model to do generation.
        config (GPT2Config): configuration of given GPT2 model.
        tokenizer (GPT2Tokenizer): if choose to use input_str parameter in self.generate(), a tokenizer is compulsory.
        generate_length (int): number of tokens which should be generated. Default: 1.
        topk_num (int): number of k in Top-k Sampling, 0 for no condition constrained,
                        equivalent to k = self.vocab_size. Default:0
        topp_prob (float): probability parameter of Top-p sampling.
                           if p = 1.0, it equals to do nothing. (nucleus sampling). Default: 1.0
        temperature (float): temperature for Top-k sampling. Default: 1.0
        min_tokens_to_keep (int): guarantee for there is at least min_tokens_to_keep token(s) generated. Default:1
        early_stop (bool): whether stop when the model generates <EOS> token.
                           It is functioned when batch_size is 1. Default: False
        return_ids (bool): whether return ids generated from Sample. Default: False
        return_last_token_logits (bool): whether return logits of last token for each time step during generation.
                                         Default: False
        append_eos (bool): whether append <EOS> token id to input_ids pass directly to GPT2Model class. Default: False

    """

    def __init__(self,
                 decoder,
                 config=None,
                 batch_size=None,
                 tokenizer=None,
                 generate_length=1,
                 topk_num=0,
                 topp_prob=1.0,
                 temperature=1.0,
                 min_tokens_to_keep=1,
                 early_stop=False,
                 return_ids=False,
                 return_last_token_logits=False,
                 append_eos=False,
                 ):

        assert config is not None, 'Config is a must for sampling.'

        self.decoder = decoder
        self.config = config
        self.tokenizer = tokenizer
        self.generate_length = generate_length
        self.topk_num = topk_num
        self.topp_prob = topp_prob
        self.temperature = temperature
        self.min_tokens_to_keep = min_tokens_to_keep
        self.early_stop = early_stop
        self.return_ids = return_ids
        self.return_last_token_logits = return_last_token_logits
        self.append_eos = append_eos

        self.seq_length = config.seq_length
        self.batch_size = config.batch_size if batch_size is None else batch_size
        self.vocab_size = config.vocab_size

        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reshape = P.Reshape()
        self.cumsum = P.CumSum()
        self.onehot = P.OneHot()
        self.cast = P.Cast()
        self.concat = P.Concat()
        self.sample_function = P.RandomCategorical(mstype.int32)
        self.filter_distribution = TopKTopP_Filter(batch_size=self.batch_size,
                                                   vocab_size=self.vocab_size,
                                                   k=self.topk_num,
                                                   p=self.topp_prob,
                                                   temperature=self.temperature,
                                                   min_tokens_to_keep=self.min_tokens_to_keep)

        if self.tokenizer is not None:
            self.eos_id = self.tokenizer.eos_token_id
        else:
            self.eos_id = config.vocab_size - 1

        if self.tokenizer is not None:
            self.eos_text = self.tokenizer.eos_token
        else:
            self.eos_text = "<|endoftext|>"

    def _extract_string_from_tensor(self, input_ids, mode="pair"):
        """
        Args:
            input_ids(Tensor): input sentences with shape [self.batch_size, self.seq_len]
            mode (str): ["pair", "single"]
                        "pair" for tasks with paired inputs `<bos> A <eos> B <eos>`,
                        such as summarization task, the dataset format `<bos> Article <eos> Summary <eos>`,
                        reading comprehension task, the dataset format `<bos> Passage Question <eos> Answer <eos>`.

                        "single" for tasks with single input `<bos> A <eos>`, such as Language Modeling, Lambada task.
        Returns:
            source_list (list): the list of source_text or first part of text.
            target_list (list): the list of target_text or second part of text.
            If self.batch_size is 1, it will return the first sentence of list, that is to say, the string.

        """
        assert self.tokenizer is not None, 'There is no tokenizer'
        source_list = [""] * self.batch_size
        target_list = [""] * self.batch_size
        eos_text = self.tokenizer.eos_token
        len_eos_text = len(eos_text)
        input_ids = self.reshape(input_ids, (self.batch_size, self.seq_length))

        if mode == "pair":
            for batch_idx in range(self.batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()[1:]

                sentence = self.tokenizer.decode(sentence_list)
                source_start = 0
                source_end = sentence.find(eos_text, 0)
                target_start = source_end + len_eos_text
                target_end = sentence[target_start:].find(eos_text, 0) + target_start
                source_list[batch_idx] = sentence[source_start:source_end]
                target_list[batch_idx] = sentence[target_start:target_end]

            return source_list, target_list

        if mode == "single":
            for batch_idx in range(self.batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()[1:]

                sentence = self.tokenizer.decode(sentence_list)
                source_start = 0
                source_end = sentence.find(eos_text, 0)
                source_list[batch_idx] = sentence[source_start:source_end]
        else:
            raise ValueError('mode:{} not supported, only support [pair, single].'.format(mode))

        return source_list

    def _tensorize_ids_with_masks(self, src_str):
        """
        Transform from string to tensor

        Args:
            src_str: string or list of strings
        Return:
            input_ids (Tensor): shape with [self.batch_size, self.seq_length]
            input_mask (Tensor): shape with [self.batch_size, self.seq_length]
            src_len_list (list): the length of tokens of src_string after decoded by self.tokenzier
        """

        if isinstance(src_str, str):
            src_str = [src_str]

        single_sentence_shape = (1, self.seq_length)
        src_len_list = list()
        input_ids = None
        input_mask = None

        for batch_idx in range(self.batch_size):
            src_ids_list = self.tokenizer.encode(src_str[batch_idx])
            src_ids_len = len(src_ids_list)
            if src_ids_len > self.seq_length:
                src_ids_list = src_ids_list[:self.seq_length]
                src_ids_len = self.seq_length

            src_len_list.append(src_ids_len)
            return_dict = self.tokenizer.prepare_for_model(src_ids_list,
                                                           max_length=self.config.seq_length,
                                                           add_special_tokens=False)

            input_ids_list = return_dict['input_ids']
            input_mask_list = return_dict['attention_mask']

            input_ids_tensor = self.reshape(Tensor(np.array(input_ids_list, dtype=int), dtype=mstype.int32),
                                            single_sentence_shape)
            input_mask_tensor = self.reshape(Tensor(np.array(input_mask_list, dtype=int), dtype=mstype.int32),
                                             single_sentence_shape)
            if batch_idx == 0:
                input_ids = input_ids_tensor
                input_mask = input_mask_tensor
            else:
                input_ids = self.concat((input_ids, input_ids_tensor))
                input_mask = self.concat((input_mask, input_mask_tensor))

        return input_ids, input_mask, src_len_list

    class LastTokenPos():
        """
        class for record input_strs and the position of their last tokens

        Args:
            input_ (Union[list, Tensor]): list if input is a list containing strings,
                                          Tensor with shape (batch_size, seq_length) representing input_mask.
        """

        def __init__(self, input_, seq_length=1024):
            if isinstance(input_, list):
                self.input_strs = input_
                self.input_mask = None
            else:
                self.input_strs = None
                self.input_mask = input_

            self.seq_length = seq_length
            if self.input_strs is not None:
                self.pos_list = [len(input_str) - 1 for input_str in self.input_strs]
            else:
                input_mask_ = P.Cast()(self.input_mask, mstype.float32)
                temp_pos_list = P.ReduceSum(keep_dims=False)(input_mask_, axis=1).asnumpy().astype(np.int32).tolist()
                # minimum value is always 0 for safety
                self.pos_list = [max(0, pos - 1) for pos in temp_pos_list]

        def get_pos(self, shift: int = 0):
            # return last token if overflow
            shift_list = [min(self.seq_length - 1, pos + shift) for pos in self.pos_list]
            return shift_list

    def _sample_from_distribution(self, distribution):
        """
        sample one token per batch from self.sample_function().

        Arg:
            distribution (Tensor): the distribution or logits of the last token of different batches.
                                   shape with [batch_size, vocab_size]

        Return:
            word_index (Tensor): shape with [batch_size, ]
        """

        distribution = self.reshape(distribution, (self.vocab_size, self.batch_size))
        topk_distribution = distribution[:self.topk_num, ::]
        topk_distribution = self.reshape(topk_distribution, (self.batch_size, -1))

        word_index = self.sample_function(P.Softmax()(topk_distribution), 1, 1)
        word_index = self.reshape(word_index, (-1,))

        return word_index

    def _input_check_and_normalize(self, input_str=None, input_ids=None, input_mask=None, generate_length=None):
        """
        input check function
        """
        if input_str is not None:
            assert self.tokenizer is not None, 'if choose to give input_str, a tokenizer is necessary.'

        if input_ids is not None:
            assert input_mask is not None, 'if input_ids is given, input_mask is required either.'

        if input_str is not None and input_ids is not None and input_mask is not None:
            print('[WARNING] Sample.generate got input_str, input_ids and input_mask, '
                  'choose input_str as default for input')

        if input_ids is None and input_mask is None:
            input_ids, input_mask, _ = self._tensorize_ids_with_masks(input_str)
        else:
            if input_str is None:
                if input_ids is not None:
                    input_str = self._extract_string_from_tensor(input_ids, mode="full")

        if generate_length is not None:
            # reload generate_length
            generate_length = int(generate_length)
            assert generate_length >= 0, 'generate_length can not be negative.'
        else:
            generate_length = self.generate_length

        return input_str, input_ids, input_mask, generate_length

    def generate(self, input_str=None, input_ids=None, input_mask=None, generate_length=None, do_sample=True):
        """
        base function for text generation given a batch_size list of str or str itself (when demo mode is on)

        Args
            input_str (list(str) or str): prompt string.
            generate_length: number of tokens to generate.

        Returns:
            generate_str: string generated by the GPT-2 model.
            full_str: input_str appended with generate_str.
        """
        input_str, input_ids, input_mask, generate_length = self._input_check_and_normalize(input_str,
                                                                                            input_ids,
                                                                                            input_mask,
                                                                                            generate_length)
        return_ids_list = [[] for i in range(self.batch_size)]

        last_token = self.LastTokenPos(input_mask, seq_length=self.seq_length)

        for i in range(generate_length):
            last_token_pos_list = last_token.get_pos(shift=i)
            early_stop_mask = [0] * self.batch_size

            # unsorted logits (distribution) of next word
            logits = self.decoder.predict(input_ids, input_mask)

            if self.return_last_token_logits is True:
                if i == 0:
                    # [batch_size, 1, vocab_size]
                    return_last_logits = extract_single_token_logits(logits, last_token_pos_list)
                else:
                    # [batch_size, 1, vocab_size] + [batch_size, i, vocab_size] --> [batch_size, i+1, vocab_size]
                    return_last_logits = P.Concat(axis=1)((return_last_logits,
                                                           extract_single_token_logits(logits, last_token_pos_list)))

            nextword_distribution = self.reshape(logits[0, last_token_pos_list[0]:last_token_pos_list[0]+1:1, ::],
                                                 (1, -1))

            # stack up nextword_distribution if batch_size is larger than 1
            if self.batch_size > 1:
                for batch_idx in range(1, self.batch_size):
                    nextword_distribution_rest = self.reshape(
                        logits[batch_idx, last_token_pos_list[batch_idx]:last_token_pos_list[batch_idx] + 1:1, ::],
                        (1, -1))
                    nextword_distribution = self.concat((nextword_distribution, nextword_distribution_rest))

            if do_sample:
                # get sampled ids
                nextword_distribution = nextword_distribution.asnumpy().astype(np.float32)
                real_next_word_index_list = self.filter_distribution.calculate(nextword_distribution)
            else:
                np_nextword_distribution = nextword_distribution.asnumpy()
                next_word_index = np.argmax(np_nextword_distribution, axis=-1)
                real_next_word_index_list = next_word_index.tolist()

            append_ids = []

            # tokenizer.decode and early_stop (if all batched generates a EOS, then it is time to say goodbye)
            for batch_idx in range(self.batch_size):
                next_word_index = real_next_word_index_list[batch_idx]
                # earlystop if the model generates a EOS token.
                if next_word_index == self.eos_id:
                    if self.early_stop:
                        early_stop_mask[batch_idx] = 1
                        if self.batch_size == 1:
                            break
                if early_stop_mask[batch_idx] == 1:
                    append_ids.append(-1)
                    continue

                return_ids_list[batch_idx].append(next_word_index)
                append_ids.append(next_word_index)

            # check early_stop mask at the end of each loop
            if 0 not in early_stop_mask or append_ids == []:
                break
            input_ids, input_mask = add_last_token(input_ids,
                                                   input_mask,
                                                   overflow_strategy="shift",
                                                   append_ids=append_ids,
                                                   next_token_pos=last_token.get_pos(shift=i + 1))

        # add str to full str
        generate_str = [""] * self.batch_size
        full_str = [""] * self.batch_size
        text_cnt = 0

        for text_ids in return_ids_list:
            text = self.tokenizer.decode(text_ids)
            generate_str[text_cnt] = text
            text_cnt += 1

        for batch_idx in range(self.batch_size):
            full_str[batch_idx] = input_str[batch_idx] + generate_str[batch_idx]

        if self.return_ids:
            if self.return_last_token_logits:
                return return_ids_list, return_last_logits
            return return_ids_list

        return generate_str, full_str
