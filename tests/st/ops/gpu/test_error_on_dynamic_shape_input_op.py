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

import pytest

from mindspore.ops.operations import _inner_ops as inner
import mindspore.context as context

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_error_on_dynamic_shape_input_is_dynamic():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    error_on_dynamic_shape_input = inner.ErrorOnDynamicShapeInput()

    with pytest.raises(ValueError) as info:
        error_on_dynamic_shape_input.infer_shape([-1])
    assert "Input is dynamically shaped" in str(info.value)

    with pytest.raises(ValueError) as info:
        error_on_dynamic_shape_input.infer_shape([1, 1, -1])
    assert "Input is dynamically shaped" in str(info.value)

    with pytest.raises(ValueError) as info:
        error_on_dynamic_shape_input.infer_shape([-1, 1, 1])
    assert "Input is dynamically shaped" in str(info.value)

    with pytest.raises(ValueError) as info:
        error_on_dynamic_shape_input.infer_shape([1, -1, 1])
    assert "Input is dynamically shaped" in str(info.value)

    with pytest.raises(ValueError) as info:
        error_on_dynamic_shape_input.infer_shape([-1, -1, -1])
    assert "Input is dynamically shaped" in str(info.value)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_error_on_dynamic_shape_input_not_dynamic():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    error_on_dynamic_shape_input = inner.ErrorOnDynamicShapeInput()
    error_on_dynamic_shape_input([1])
    error_on_dynamic_shape_input([1, 1])
    error_on_dynamic_shape_input([23, 12, 9712])
