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

"""
LlamaProcessor
"""

from mindformers.llama.tokenization import PreTrainedTokenizerBase


__all__ = ['LlamaProcessor']
class LlamaProcessor:
    """
    Llama processor,
    consists of a tokenizer (PreTrainedTokenizerBase) for text input.
    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer of LlamaModel.
        max_length (`int`, *optional*, defaults to 128):
            The maximum length (in number of tokens) for the inputs to LlamaModel.
        padding (`str`, *optional*, defaults to `max_length`):
            Activates and controls padding. Accepts the following values:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors instead of list of python integers. Acceptable values are:

            - `'np'`: Return Numpy `np.ndarray` objects.
            - `'ms'`: Return Numpy `ms.Tensor` objects.
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(self, tokenizer=None,
                 max_length=128, padding='max_length', return_tensors='ms'):
        super(LlamaProcessor, self).__init__(
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors
        )

    def __call__(self, text_input=None, image_input=None):
        """call function"""
        output = {}
        if text_input is not None and self.tokenizer:
            if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
                raise TypeError(f"tokenizer should inherited from the PreTrainedTokenizerBase,"
                                f" but got {type(self.tokenizer)}.")
            # Format the input into a batch
            if isinstance(text_input, str):
                text_input = [text_input]
            text_output = self.tokenizer(text_input, return_tensors=self.return_tensors,
                                         max_length=self.max_length,
                                         padding=self.padding)["input_ids"]
            output['text'] = text_output
        return output
