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
"""test_dataset_utils"""
import pytest

import mindspore as ms
from mindspore.train._utils import _construct_tensor_list


def test_init_dataset_graph():
    types = (ms.float32, ms.float32)
    shapes = ((1, 3, 224, 224), (32,))
    _construct_tensor_list(types, shapes)


def test_init_dataset_graph_one_dim():
    types = (ms.float32,)
    shapes = ((1, 3, 224, 224),)
    _construct_tensor_list(types, shapes)


def test_init_dataset_graph_dim_error():
    types = (ms.float32, ms.float32)
    shapes = ((1, 3, 224, 224),)
    with pytest.raises(ValueError):
        _construct_tensor_list(types, shapes)
