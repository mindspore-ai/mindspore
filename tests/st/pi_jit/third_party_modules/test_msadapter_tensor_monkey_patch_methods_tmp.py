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
# ============================================================================
"""Test msadapter Tensor monkey-patch methods"""

import pytest

import mindspore
from mindspore import jit, Tensor, ops

from tests.st.pi_jit.share.utils import assert_equal, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark


# mindspeed patch
def ensure_contiguous_wrapper(fn):
    def wrapper(tensor, *args, **kwargs):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        # LOAD_FAST    0 tensor
        # BUILD_LIST   1
        # LOAD_FAST    1 args
        # LIST_EXTEND  1 // merge tensor and *args
        return fn(tensor, *args, **kwargs)

    return wrapper


# msadapter patch
def view(self, *shape):
    result = []
    if type(shape) is tuple:
        for items in shape:
            if not isinstance(items, int):
                for item in items:
                    result.append(item)
            else:
                result.append(items)
    return ops.reshape(self, result)


@pytest.mark.skip(reason="will support it later")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_msadapter_and_mindspeed_Tensor_view_monkey_patch():
    """
    Feature: Test msadapter+mindspeed Tensor.view monkey patch method.
    Description: monkey patch Tensor.view with custom view() function.
    Expectation: no exception, no graph break.
    """

    # Monkey patch
    Tensor.view = ensure_contiguous_wrapper(view)

    def fn(x: Tensor):
        return x.view(-1, 2) + 1

    x = mindspore.tensor([[1, 2, 3, 4]])  # Shape is (1, 4)

    o1 = fn(x)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)
