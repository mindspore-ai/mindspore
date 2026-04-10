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
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_pynative_amp_with_tuple_input1():
    """
    Feature: Pynative AMP
    Description: Test pynative amp with tuple input in RunOp
    Expectation: run success
    """
    def func(a, b, c):
        return ms.ops.addn([a, b, c])


    a = ms.Tensor([1, 2], ms.float16)
    b = ms.Tensor([3, 4], ms.float16)
    c = ms.Tensor([5, 6], ms.float16)

    func = ms.amp.auto_mixed_precision(func, "auto")

    output = func(a, b, c)
    assert output.dtype == ms.float16

    a = ms.Tensor([1, 2], ms.float16)
    b = ms.Tensor([3, 4], ms.float32)
    c = ms.Tensor([5, 6], ms.float16)

    output = func(a, b, c)
    assert output.dtype == ms.float32


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_pynative_amp_with_tuple_input2():
    """
    Feature: Pynative AMP
    Description: Test pynative amp with tuple input in pyboost.
    Expectation: run success
    """
    def func(a, b, c):
        return ms.ops.concat([a, b, c])


    a = ms.Tensor([1, 2], ms.float16)
    b = ms.Tensor([3, 4], ms.float16)
    c = ms.Tensor([5, 6], ms.float16)

    func = ms.amp.auto_mixed_precision(func, "auto")

    output = func(a, b, c)
    assert output.dtype == ms.float16

    a = ms.Tensor([1, 2], ms.float16)
    b = ms.Tensor([3, 4], ms.float16)
    c = ms.Tensor([5, 6], ms.float32)

    output = func(a, b, c)
    assert output.dtype == ms.float32
