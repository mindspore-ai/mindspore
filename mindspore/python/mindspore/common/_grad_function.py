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

"""Defines custom autograd function with functional form."""

__all__ = ['_Function']

from typing import Any
from mindspore._c_expression import FunctionBase as FunctionBase_
from mindspore.common.tensor import Tensor


class _Function(FunctionBase_):
    """
    A Class provides the ability to custom autograd function. The api refers
    to the following files from pytorch：
    https://github.com/pytorch/pytorch/blob/main/torch/autograd/function.py

    Note:
        It is only supported in pynative mode.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwars: Any) -> Any:
        raise NotImplementedError("forward function should be customized.")

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError("backward function should be customized.")

    @classmethod
    def apply(cls, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def save_for_backward(self, *tensors: Tensor):
        self.saved_tensors = tensors

    def mark_dirty(self, *args: Tensor):
        self.dirty_tensors = args

    def mark_non_differentiable(self, *args: Tensor):
        self.non_differentiable = args

    def set_materialize_grads(self, value: bool):
        self.materialize_grads = value
