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
""" test nn embedding """
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.common import dtype
from mindspore.common.api import _executor
from mindspore.nn import Embedding, MultiFieldEmbeddingLookup
from ..ut_filter import non_graph_engine


@non_graph_engine
def test_check_embedding_1():
    net = Embedding(20000, 768, False)
    input_data = Tensor(np.ones([8, 128]), dtype.int32)
    _executor.compile(net, input_data)


@non_graph_engine
def test_check_embedding_2():
    net = Embedding(20000, 768, True)
    input_data = Tensor(np.ones([8, 128]), dtype.int32)
    _executor.compile(net, input_data)


@non_graph_engine
def test_check_embedding_3():
    net = Embedding(20000, 768, True, "zeros")
    input_data = Tensor(np.ones([8, 128]), dtype.int32)
    _executor.compile(net, input_data)


def compile_multi_field_embedding(shape_id, shape_value, shape_field,
                                  type_id, type_value, type_field):
    net = MultiFieldEmbeddingLookup(20000, 768, 3)
    input_data = Tensor(np.ones(shape_id), type_id)
    input_value = Tensor(np.ones(shape_value), type_value)
    input_field = Tensor(np.ones(shape_field), type_field)
    _executor.compile(net, input_data, input_value, input_field)


@non_graph_engine
def test_check_multifield_embedding_right_type():
    compile_multi_field_embedding((8, 200), (8, 200), (8, 200),
                                  dtype.int64, dtype.float32, dtype.int32)


@non_graph_engine
def test_check_multifield_embedding_false_type_input():
    with pytest.raises(TypeError):
        compile_multi_field_embedding((8, 200), (8, 200), (8, 200),
                                      dtype.int16, dtype.float32, dtype.int32)


@non_graph_engine
def test_check_multifield_embedding_false_type_value():
    with pytest.raises(TypeError):
        compile_multi_field_embedding((8, 200), (8, 200), (8, 200),
                                      dtype.int16, dtype.float16, dtype.int32)


@non_graph_engine
def test_check_multifield_embedding_false_type_field_id():
    with pytest.raises(TypeError):
        compile_multi_field_embedding((8, 200), (8, 200), (8, 200),
                                      dtype.int16, dtype.float32, dtype.int16)


@non_graph_engine
def test_check_multifield_embedding_false_input_shape():
    with pytest.raises(ValueError):
        compile_multi_field_embedding((8,), (8, 200), (8, 200),
                                      dtype.int16, dtype.float32, dtype.int16)


@non_graph_engine
def test_check_multifield_embedding_false_value_shape():
    with pytest.raises(ValueError):
        compile_multi_field_embedding((8, 200), (8,), (8, 200),
                                      dtype.int16, dtype.float32, dtype.int16)

@non_graph_engine
def test_print_embedding():
    net = Embedding(20000, 768, False)
    print(net)
