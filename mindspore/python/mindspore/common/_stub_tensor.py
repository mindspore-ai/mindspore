# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Stub Tensor implementation."""

from mindspore.common.tensor import Tensor
from mindspore.common.api import _pynative_executor, _convert_python_data


class StubTensor(Tensor):
    """stub tensor for async op run."""

    def __init__(self, stub):
        Tensor.__init__(self, internal=True)
        self.stub = stub
        self.tensor = None

    def __repr__(self):
        return self.data().__repr__()

    def __str__(self):
        return self.data().__str__()

    @property
    def shape(self):
        """shape stub."""
        if self.tensor:
            return self.tensor.shape
        return _pynative_executor.get_stub_shape(self.stub)

    @property
    def dtype(self):
        """dtype stub."""
        if self.tensor:
            return self.tensor.dtype
        return _pynative_executor.get_stub_dtype(self.stub)

    def data(self):
        """get real tensor data."""
        if self.tensor is None:
            val = _pynative_executor.get_stub_value(self.stub)
            self.tensor = _convert_python_data(val)
        return self.tensor

    def asnumpy(self):
        """api stub."""
        return self.data().asnumpy()


_stub_map = [
    StubTensor
]


def _convert_stub(stub_type, stub):
    """convert stub node to stub tensor."""
    return _stub_map[stub_type](stub)
