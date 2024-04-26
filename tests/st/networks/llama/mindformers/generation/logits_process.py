# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Logits Processor for generation."""
import inspect
from threading import Thread
from typing import Union, List

import numpy as np

from .utils import log_softmax, softmax, topk


__all__ = ["LogitsProcessor", "LogitsWarper", "LogitsProcessorList", "RepetitionPenaltyLogitsProcessor",
           "LogitNormalization", "TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper",
           "MinLengthLogitsProcessor", "MinNewTokensLengthLogitsProcessor"]


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids, scores):
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation
    with multinomial sampling."""

    def __call__(self, input_ids, scores):
        """Torch method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently
    process a `scores` input tensor. This class inherits from list and adds a specific *__call__* method
    to apply each [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids, scores, is_finished=None, **kwargs):
        all_threads = []
        for i in range(0, input_ids.shape[0]):
            if is_finished and is_finished[i]:
                continue
            thread = Thread(target=self.process,
                            args=(i, input_ids, scores), kwargs=kwargs)
            all_threads.append(thread)
            thread.start()
        for thread in all_threads:
            thread.join()
        return scores

    def process(self, i, input_ids, scores, **kwargs):
        """apply process"""
        input_ids = input_ids[i:i + 1]
        scores_i = scores[i:i + 1]
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores_i = processor(input_ids, scores_i, **kwargs)
            else:
                scores_i = processor(input_ids, scores_i)
        scores[i] = scores_i


class TemperatureLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError(
                f"`temperature` has to be a strictly positive float, but is {temperature}"
            )

        self.temperature = temperature

    def __call__(self, input_ids, scores):
        scores = scores / self.temperature
        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, repetition_penalty: float):
        repetition_penalty = float(repetition_penalty)
        if repetition_penalty <= 0:
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {repetition_penalty}"
            )

        self.penalty = repetition_penalty

    def __call__(self, input_ids, scores):
        score = np.take_along_axis(scores, input_ids, axis=1)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        negative_index = score < 0
        positive_index = ~negative_index
        score[negative_index] = score[negative_index] * self.penalty
        score[positive_index] = score[positive_index] / self.penalty

        np.put_along_axis(scores, input_ids, score, axis=1)
        return scores


class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-50000`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
        candidate_token_num (`int`, *optional*, defaults to 200):
            Number of candidate tokens to calculate top_p. this can avoid sorting a huge seq,
            save time to speed up generation.
    """

    def __init__(self, top_p: float, filter_value: float = -50000, min_tokens_to_keep: int = 1,
                 candidate_token_num: int = 200):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 0):
            raise ValueError(
                f"`min_tokens_to_keep` has to be a non-negative integer, but is {min_tokens_to_keep}"
            )

        self.top_p = top_p
        self.filter_value = float(filter_value)
        self.min_tokens_to_keep = min_tokens_to_keep
        self.candicate_token_num = candidate_token_num

    def __call__(self, input_ids, scores):
        candidate_logits, candidate_indices = topk(scores, self.candicate_token_num)
        cumulative_probs = softmax(candidate_logits)
        cumulative_probs = np.cumsum(cumulative_probs, axis=-1)
        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_keep = cumulative_probs < self.top_p
        # add the last token that exceed top_p
        sorted_indices_to_keep = np.concatenate(
            [np.ones(shape=(scores.shape[0], 1)).astype(np.bool_), sorted_indices_to_keep[..., :-1]],
            axis=-1
        )
        # Keep at least min_tokens_to_keep
        sorted_indices_to_keep[..., :self.min_tokens_to_keep] = 1

        # set remove indices, filter negative value
        indices_to_remove = np.ones_like(scores).astype(np.bool_)
        np.put_along_axis(
            indices_to_remove, candidate_indices, ~sorted_indices_to_keep, axis=-1
        )
        scores[indices_to_remove] = self.filter_value

        return scores


class TopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -50000, min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"`top_k` has to be a strictly positive integer, but is {top_k}"
            )

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = float(filter_value)

    def __call__(self, input_ids, scores: np.ndarray):
        top_k = min(self.top_k, scores.shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < topk(scores, top_k)[0][..., -1, None]
        scores[indices_to_remove] = self.filter_value
        return scores


class LogitNormalization(LogitsProcessor, LogitsWarper):
    r"""
    [`LogitsWarper`] and [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.
    """

    def __call__(self, input_ids, scores):
        scores = log_softmax(scores, axis=-1)
        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, min_length: int, eos_token_id: Union[int, List[int]], pad_token_id: int):
        min_length = int(min_length)
        if min_length < 0:
            raise ValueError(f"`min_length` has to be a non-negative integer, but is {min_length}")

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        self.min_length = min_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]

        valid_length_each_example = []
        for i in range(batch_size):
            valid_length_each_example.append(
                np.max(np.argwhere(input_ids[i] != self.pad_token_id))
                + 1
            )
        valid_length_each_example = np.array(valid_length_each_example)

        cur_len = np.max(valid_length_each_example)
        if cur_len < self.min_length:
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")
        return scores


class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
    Note that for decoder-only models, such as Llama2, `min_length` will compute the length of `prompt + newly
    generated tokens` whereas for other models it will behave as `min_new_tokens`, that is, taking only into account
    the newly generated ones.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the
            input length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, prompt_length_to_skip: int, min_new_tokens: int, eos_token_id: Union[int, List[int]],
                 pad_token_id: int):
        for arg_name, arg_value in \
                [("prompt_length_to_skip", prompt_length_to_skip), ("min_new_tokens", min_new_tokens)]:
            arg_value = int(arg_value)
            if arg_value < 0:
                raise ValueError(f"`{arg_name}` has to be a positive integer, but is {arg_value}")

            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]

            self.prompt_length_to_skip = prompt_length_to_skip
            self.min_new_tokens = min_new_tokens
            self.eos_token_id = eos_token_id
            self.pad_token_id = pad_token_id

    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]

        valid_length_each_example = []
        for i in range(batch_size):
            valid_length_each_example.append(
                np.max(np.argwhere(input_ids[i] != self.pad_token_id))
                + 1
            )
        valid_length_each_example = np.array(valid_length_each_example)

        cur_len = np.max(valid_length_each_example)
        new_tokens_length = cur_len - self.prompt_length_to_skip
        if new_tokens_length < self.min_new_tokens:
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")

        return scores
