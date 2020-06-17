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

"""
mindspore.dataset.text
"""
import platform
from .transforms import Lookup, JiebaTokenizer, UnicodeCharTokenizer, Ngram, WordpieceTokenizer, TruncateSequencePair
from .utils import to_str, to_bytes, JiebaMode, Vocab, NormalizeForm

__all__ = [
    "Lookup", "JiebaTokenizer", "UnicodeCharTokenizer", "Ngram",
    "to_str", "to_bytes", "JiebaMode", "Vocab", "WordpieceTokenizer", "TruncateSequencePair"
]

if platform.system().lower() != 'windows':
    from .transforms import UnicodeScriptTokenizer, WhitespaceTokenizer, CaseFold, NormalizeUTF8, \
        RegexReplace, RegexTokenizer, BasicTokenizer, BertTokenizer

    __all__.append(["UnicodeScriptTokenizer", "WhitespaceTokenizer", "CaseFold", "NormalizeUTF8",
                    "RegexReplace", "RegexTokenizer", "BasicTokenizer", "BertTokenizer", "NormalizeForm"])
