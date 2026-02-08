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
import mindspore.context as context
import pytest
from mindspore.graph.api import _pynative_executor
from mindspore.ops.operations import _inner_ops as inner

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_error_on_dynamic_shape_input_is_dynamic():
    """
    Feature: Test dynamic shape input exception.
    Description: Test dynamic shape input exception.
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    error_on_dynamic_shape_input = inner.ErrorOnDynamicShapeInput()

    with pytest.raises(ValueError) as info:
        error_on_dynamic_shape_input.infer_shape([-1])
        _pynative_executor.sync()
    assert "Input is dynamically shaped" in str(info.value)

    with pytest.raises(ValueError) as info:
        error_on_dynamic_shape_input.infer_shape([1, 1, -1])
        _pynative_executor.sync()
    assert "Input is dynamically shaped" in str(info.value)

    with pytest.raises(ValueError) as info:
        error_on_dynamic_shape_input.infer_shape([-1, 1, 1])
        _pynative_executor.sync()
    assert "Input is dynamically shaped" in str(info.value)

    with pytest.raises(ValueError) as info:
        error_on_dynamic_shape_input.infer_shape([1, -1, 1])
        _pynative_executor.sync()
    assert "Input is dynamically shaped" in str(info.value)

    with pytest.raises(ValueError) as info:
        error_on_dynamic_shape_input.infer_shape([-1, -1, -1])
        _pynative_executor.sync()
    assert "Input is dynamically shaped" in str(info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_error_on_dynamic_shape_input_not_dynamic():
    """
    Feature: Test dynamic shape input exception.
    Description: Test dynamic shape input exception.
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    error_on_dynamic_shape_input = inner.ErrorOnDynamicShapeInput()
    error_on_dynamic_shape_input([1])
    error_on_dynamic_shape_input([1, 1])
    error_on_dynamic_shape_input([23, 12, 9712])
