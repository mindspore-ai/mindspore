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
"""Test view + inplace"""
import numpy as np
import pytest
import mindspore as ms
from mindspore.graph.api import _pynative_executor
from mindspore import Tensor
from tests.mark_utils import arg_mark


class TestViewInplace:
    def is_shared_address(self, input, other): # pylint: disable=redefined-builtin
        if input.device != other.device:
            return False
        return input.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()

    def is_view_of(self, base, other):
        return other is not base and self.is_shared_address(base, other)

    def is_view_of_same_base(self, base, input, other): # pylint: disable=redefined-builtin
        if other is input:
            return False
        return self.is_view_of(base, input) and self.is_view_of(base, other)

    def generate_input(self, shape):
        input_np = np.random.rand(*shape)
        return input_np

    def single_view_test(self, shape):
        input_np = self.generate_input(shape)
        assert len(shape) >= 2
        perm = list(range(len(shape)))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        expect_output = input_np.transpose(perm)

        input_ms = Tensor(input_np).to("Ascend")
        output = input_ms.transpose(-1, -2)

        assert self.is_view_of(input_ms, output)
        assert np.allclose(output.asnumpy(), expect_output)

    def multi_view_test(self, shape):
        input_np = self.generate_input(shape)
        assert len(shape) >= 2
        expect_output0 = input_np[-1]
        expect_output1 = expect_output0[-1]

        input_ms = Tensor(input_np).to("Ascend")
        ouptut0 = input_ms.select(0, -1)
        output1 = ouptut0.select(0, -1)

        assert self.is_view_of_same_base(input_ms, ouptut0, output1)
        assert np.allclose(ouptut0.asnumpy(), expect_output0)
        assert np.allclose(output1.asnumpy(), expect_output1)

    def non_view_test(self, shape):
        input_np = self.generate_input(shape)
        expect_output = input_np.T.reshape((-1,))

        input_ms = Tensor(input_np).to("Ascend")
        output = input_ms.T.reshape(-1)

        assert not self.is_view_of(input_ms, output)
        assert np.allclose(output.asnumpy(), expect_output)

    def clone_with_view_test(self, shape):
        input_np = self.generate_input(shape)
        expect_output = input_np.T

        # clone + view
        input_ms = Tensor(input_np).to("Ascend")
        output = input_ms.clone().T

        assert not self.is_shared_address(input_ms, output)
        assert np.allclose(output.asnumpy(), expect_output)

        # view + clone
        output_view = input_ms.T
        output = output_view.clone()

        assert not self.is_shared_address(input_ms, output)
        assert np.allclose(output.asnumpy(), expect_output)

    def single_inplace_test(self, shape):
        input_np = self.generate_input(shape)
        expect_output = input_np + 1

        input_ms = Tensor(input_np).to("Ascend")
        input_ms.add_(1)

        assert np.allclose(input_ms.asnumpy(), expect_output)

    def multi_inplace_test(self, shape):
        input_np = self.generate_input(shape)
        expect_output = input_np + 1 + 2.5

        input_ms = Tensor(input_np).to("Ascend")
        input_ms.add_(1)
        input_ms.add_(2.5)

        assert np.allclose(input_ms.asnumpy(), expect_output)

    def clone_with_inplace_test(self, shape):
        input_np = self.generate_input(shape)
        expect_output = input_np + 1

        # clone + inplace
        input_ms = Tensor(input_np).to("Ascend")
        output = input_ms.clone()
        output.add_(1)

        assert not self.is_shared_address(input_ms, output)
        assert np.allclose(output.asnumpy(), expect_output)

        # inplace + clone
        input_ms.add_(1)
        output = input_ms.clone()
        assert not self.is_shared_address(input_ms, output)
        assert np.allclose(output.asnumpy(), expect_output)

    def single_view_inplace_test(self, shape):
        input_np = self.generate_input(shape)

        # view + inplace
        expect_output = input_np.T + 2.5

        input_ms = Tensor(input_np).to("Ascend")
        output = input_ms.T
        output.add_(2.5)

        assert self.is_shared_address(input_ms, output)
        assert np.allclose(output.asnumpy(), expect_output)

        # inplace + view
        expect_output = (input_np + 2.5).T

        input_ms = Tensor(input_np).to("Ascend")
        input_ms.add_(2.5)
        output = input_ms.T

        assert self.is_view_of(output, input_ms)
        assert np.allclose(output.asnumpy(), expect_output)

    def multi_view_inplace_test(self, shape):
        input_np = self.generate_input(shape)
        expect_output = (input_np[-1] + 10)[-1] + 20

        input_ms = Tensor(input_np).to("Ascend")
        output0 = input_ms[-1]
        output0.add_(10)
        output1 = output0[-1]
        output1.add_(20)

        assert self.is_view_of_same_base(input_ms, output0, output1)
        assert self.is_shared_address(output0, output1)
        assert np.allclose(output1.asnumpy(), expect_output)

    def set_data_from_view_tensor_test(self):
        input_np = np.random.rand(2, 2)
        expect_np = input_np.transpose((1, 0))

        x = ms.Tensor(input_np)
        y = x.transpose(-1, -2)
        x.data = y
        assert np.allclose(x.asnumpy(), expect_np)

    def inplace_with_interneal_overlap_test(self):
        with pytest.raises(RuntimeError,
                           match="This tensor has multi element reference " \
                           "to the same memory address,which is forbidden"):
            input_ms = ms.Tensor(np.random.rand(1, 2)).to("Ascend")
            view_tensor = input_ms.expand(4, 2)
            view_tensor.add_(1)
            _pynative_executor.sync()

    def inplace_with_partial_overlap_test(self):
        with pytest.raises(RuntimeError,
                           match=r"Unsupported operations: some elements of the input tensor and " \
                           r"the written-to tensor refer to a single memory location\. " \
                           r"Please clone\(\) the tensor before performing the operation\."):
            input_ms = ms.Tensor(np.random.rand(4, 4)).to("Ascend")
            view0 = input_ms[1:]
            view1 = input_ms[:-1]
            view0.copy_(view1)
            _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
def test_view_inplace():
    """
    Feature: test pyboost view and inplace
    Description: test view and inplace by pyboost
    Expectation: success
    """
    ms.set_device("Ascend")
    tester = TestViewInplace()

    for shape in [(4, 8), (20, 10, 20), (2, 2, 4, 16, 8)]:
        tester.single_view_test(shape)
        tester.multi_view_test(shape)
        tester.non_view_test(shape)
        tester.clone_with_view_test(shape)

        tester.single_inplace_test(shape)
        tester.multi_inplace_test(shape)
        tester.clone_with_inplace_test(shape)

        tester.single_view_inplace_test(shape)
        tester.multi_view_inplace_test(shape)

    tester.set_data_from_view_tensor_test()
    tester.inplace_with_interneal_overlap_test()
    tester.inplace_with_partial_overlap_test()
