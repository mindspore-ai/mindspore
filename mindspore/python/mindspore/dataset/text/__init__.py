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

See `Text Transforms
<https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html#text-transforms>`_ tutorial for more details.

Descriptions of common data processing terms are as follows:

- TensorOperation, the base class of all data processing operations implemented in C++.
- TextTensorOperation, the base class of all text processing operations. It is a derived class of TensorOperation.

The data transform operation can be executed in the data processing pipeline or in the eager mode:

- Pipeline mode is generally used to process big datasets. Examples refer to
  `introduction to data processing pipeline <https://www.mindspore.cn/docs/en/master/api_python/
  mindspore.dataset.html#introduction-to-data-processing-pipeline>`_ .
- Eager mode is more like a function call to process data. Examples refer to
  `Lightweight Data Processing <https://www.mindspore.cn/tutorials/en/master/advanced/dataset/eager.html>`_ .
"""
import platform

from . import transforms
from . import utils
from .transforms import AddToken, JiebaTokenizer, Lookup, Ngram, PythonTokenizer, SentencePieceTokenizer, \
    SlidingWindow, ToNumber, ToVectors, Truncate, TruncateSequencePair, UnicodeCharTokenizer, WordpieceTokenizer
from .utils import CharNGram, FastText, GloVe, JiebaMode, NormalizeForm, SentencePieceModel, SentencePieceVocab, \
    SPieceTokenizerLoadType, SPieceTokenizerOutType, Vectors, Vocab, to_bytes, to_str

if platform.system().lower() != 'windows':
    from .transforms import BasicTokenizer, BertTokenizer, CaseFold, FilterWikipediaXML, NormalizeUTF8, RegexReplace, \
        RegexTokenizer, UnicodeScriptTokenizer, WhitespaceTokenizer
