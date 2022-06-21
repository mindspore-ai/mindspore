# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""
This module is to support text processing for NLP. It includes two parts:
text transforms and utils. text transforms is a high performance
NLP text processing module which is developed with ICU4C and cppjieba.
utils provides some general methods for NLP text processing.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.text as text

Descriptions of common data processing terms are as follows:

- TensorOperation, the base class of all data processing operations implemented in C++.
- TextTensorOperation, the base class of all text processing operations. It is a derived class of TensorOperation.
"""
import platform

from . import transforms
from . import utils
from .transforms import JiebaTokenizer, Lookup, Ngram, PythonTokenizer, SentencePieceTokenizer, SlidingWindow, \
    ToNumber, ToVectors, TruncateSequencePair, UnicodeCharTokenizer, WordpieceTokenizer
from .utils import CharNGram, FastText, GloVe, JiebaMode, NormalizeForm, SentencePieceModel, SentencePieceVocab, \
    SPieceTokenizerLoadType, SPieceTokenizerOutType, Vectors, Vocab, to_bytes, to_str

if platform.system().lower() != 'windows':
    from .transforms import BasicTokenizer, BertTokenizer, CaseFold, FilterWikipediaXML, NormalizeUTF8, RegexReplace, \
        RegexTokenizer, UnicodeScriptTokenizer, WhitespaceTokenizer
