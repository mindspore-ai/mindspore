# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import context, jit
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_key_is_type():
    """
    Feature: dict key.
    Description: support key of dict is type.
    Expectation: No exception.
    """
    @jit
    def dict_key_is_type():
        return ms.dtype_to_nptype(ms.float32)

    out = dict_key_is_type()
    assert str(out) == "<class 'numpy.float32'>"


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dict_key_is_none():
    """
    Feature: dict key.
    Description: support key of dict is none.
    Expectation: No exception.
    """
    @jit
    def dict_key_is_none():
        return {None: 1}

    out = dict_key_is_none()
    assert out == {None: 1}
