# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""text transform - lookup"""

import numpy as np
import pytest
import mindspore
import mindspore.dataset as ds
from mindspore.dataset import text


TEST_DATA_DATASET_FUNC ="../data/dataset/"


VOCAB_FILE = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testVocab/vocab_file.txt"
DATA_FILE = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testVocab/words.txt"


def test_lookup_operation_01():
    """
    Feature: Lookup op
    Description: Test Lookup op with vocab from list and different data types
    Expectation: Successfully convert words to their corresponding vocabulary IDs
    """
    # Test vocab from list
    vocab = text.Vocab.from_list(["?", "##", "with", "the", "test", "符号"])
    lookup = text.Lookup(vocab=vocab, unknown_token="test")
    data = "with"
    res = lookup(data)
    assert res == 2

    # Test vocab from list
    vocab = text.Vocab.from_list(["?", "##", "with", "the", "test", "符号"])
    lookup = text.Lookup(vocab=vocab, unknown_token="test")
    data = "符号"
    res = lookup(data)
    assert res == 5

    # Test vocab from list
    vocab = text.Vocab.from_list("keyboard-interactive client the With by password".split(" "))
    lookup = text.Lookup(vocab=vocab, data_type=mindspore.int32)
    data = "client"
    res = lookup(data)
    assert res == 1

    # Test vocab from list
    vocab = text.Vocab.from_list("keyboard-interactive client the With by password".split(" "))
    lookup = text.Lookup(vocab=vocab, data_type=mindspore.float32)
    data = "client"
    res = lookup(data)
    assert res == 1.0


def test_lookup_exception_01():
    """
    Feature: Lookup op
    Description: Test Lookup op with invalid parameters and vocab types
    Expectation: Raise expected exceptions for invalid inputs
    """
    # Test lookup without parameters
    with pytest.raises(TypeError, match="missing a required argument: 'vocab'"):
        text.Lookup()

    # Test lookup with vocab is int
    with pytest.raises(TypeError, match="vocab is not an instance of text.Vocab"):
        text.Lookup(vocab=6)

    # Test lookup with only input unknown
    with pytest.raises(TypeError, match="missing a required argument: 'vocab'"):
        text.Lookup(unknown_token=3)

    # Test lookup with input unknown is negative
    vocab = text.Vocab.from_list("keyboard-interactive client the With by password".split(" "))
    with pytest.raises(TypeError, match=r"Argument unknown_token with value -1 is not of type \[\<class \'str\'\>\]."):
        text.Lookup(vocab, -1)

    # Test lookup with input unknown=0
    vocab = text.Vocab.from_list("With the allowed by password".split(" "))
    with pytest.raises(TypeError, match=r"Argument unknown_token with value 0 is not of type \[\<class \'str\'\>\]."):
        lookup = text.Lookup(vocab, 0)
        data = ds.TextFileDataset(DATA_FILE, shuffle=False)
        data = data.map(operations=lookup, input_columns=["text"])
        ind = 0
        res = [0, 0, 2, 3, 1, 0, 4, 0]
        for d in data.create_dict_iterator(output_numpy=True):
            assert d["text"] == res[ind], ind
            ind += 1

    # Test lookup with input unknown=1
    vocab = text.Vocab.from_list("With the allowed by password".split(" "))
    with pytest.raises(TypeError, match=r"Argument unknown_token with value 1 is not of type \[\<class \'str\'\>\]."):
        lookup = text.Lookup(vocab, 1)
        data = ds.TextFileDataset(DATA_FILE, shuffle=False)
        data = data.map(operations=lookup, input_columns=["text"])
        ind = 0
        for _ in data.create_dict_iterator(output_numpy=True):
            ind += 1

    # Test lookup with input unknown=0
    vocab = text.Vocab.from_list("With the allowed by password".split(" "))
    with pytest.raises(TypeError, match=r"Argument unknown_token with value 0 is not of type \[\<class \'str\'\>\]."):
        lookup = text.Lookup(vocab, 0)
        data = ds.TextFileDataset(DATA_FILE, shuffle=False)
        data = data.map(operations=lookup, input_columns=["text"])
        ind = 0
        res = [0, 2, 2, 3, 1, 2, 4, 2]
        for d in data.create_dict_iterator(output_numpy=True):
            assert d["text"] == res[ind]
            ind += 1

    # Test vocab from list,dtype is not mstype
    vocab = text.Vocab.from_list("keyboard-interactive client the With by password".split(" "))
    with pytest.raises(TypeError, match=r"Argument data_type with value \<class \'numpy.float32\'\> is not of "
                                        r"type \[\<class \'mindspore._c_expression.typing.Type\'\>\]."):
        _ = text.Lookup(vocab=vocab, data_type=np.float32)

    # Test vocab from list
    vocab = text.Vocab.from_list(["?", "##", "with", "the", "test", "符号"])
    lookup = text.Lookup(vocab=vocab, unknown_token="test")
    data = 1
    with pytest.raises(RuntimeError, match="Lookup: input is not string datatype"):
        _ = lookup(data)
