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
"""import"""
import sys
import os
import six
from src.common.rule import MaxTruncation

try:
    import pkg_resources
    get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                                os.path.join(*res))
except ImportError:
    get_module_res = lambda *res: open(os.path.normpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__), *res)), 'rb')

PY2 = sys.version_info[0] == 2

default_encoding = sys.getfilesystemencoding()

if PY2:
    text_type = unicode
    string_types = (str, unicode)

    iterkeys = lambda d: d.iterkeys()
    itervalues = lambda d: d.itervalues()
    iteritems = lambda d: d.iteritems()

else:
    text_type = str
    string_types = (str,)
    xrange = range

    iterkeys = lambda d: iter(d.keys())
    itervalues = lambda d: iter(d.values())
    iteritems = lambda d: iter(d.items())





def truncation_words(words, max_seq_length, truncation_type):
    """
    :param words:
    :param max_seq_length:
    :param truncation_type:
    :return:
    """
    if len(words) > max_seq_length:
        if truncation_type == MaxTruncation.KEEP_HEAD:
            words = words[0: max_seq_length]
        elif truncation_type == MaxTruncation.KEEP_TAIL:
            tmp = words[0: max_seq_length - 1]
            tmp.append(words[-1])
            words = tmp
        elif truncation_type == MaxTruncation.KEEP_BOTH_HEAD_TAIL:
            tmp = words[1: max_seq_length - 2]
            tmp.insert(0, words[0])
            tmp.insert(max_seq_length - 1, words[-1])
            words = tmp
        else:
            words = words[0: max_seq_length]

    return words


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        if isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
    return text
