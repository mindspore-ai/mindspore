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
"""Silent Check."""
import os

from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype

from . import operations
from .operations._inner_ops import _MirrorSilentCheck
from .operations import RmsNorm as OriginRmsNorm
from .operations import LayerNorm as OriginLayerNorm
from .primitive import Primitive


NPU_ASD_ENABLE = 'NPU_ASD_ENABLE'



class ASDBase:
    """
    ASD Base Class.
    """
    _index = 0
    __ms_class__ = True

    def __init__(self, cls, *args, **kwargs):
        self.op = cls(*args, **kwargs)
        self.check_op = _MirrorSilentCheck()
        self._suffix = "ASD_" + cls.__name__
        primitive_attr = dir(Primitive)
        self._op_attr_dict = {
            name for name in primitive_attr if not name.startswith("_")}
        self.enable_check = os.environ.get(NPU_ASD_ENABLE) == "1"

    def __getattr__(self, name):
        def method_wrapper(*args, **kwargs):
            out = getattr(self.op, name)(*args, **kwargs)
            if out is self.op:
                return self
            return out

        if name in self._op_attr_dict:
            if callable(getattr(self.op, name)):
                return method_wrapper
        if hasattr(self.op, name):
            return getattr(self.op, name)
        return super().__getattr__(self, name)

    def __repr__(self):
        return self.op.__repr__()

    def generate_params(self):
        """
        Generate check params.
        """
        pre_val = Parameter(Tensor(0, mstype.float32),
                            name=f"{self._suffix}_pre_val_{self._index}",
                            requires_grad=False)
        min_val = Parameter(Tensor(0, mstype.float32),
                            name=f"{self._suffix}_min_val_{self._index}",
                            requires_grad=False)
        max_val = Parameter(Tensor(0, mstype.float32),
                            name=f"{self._suffix}_max_val_{self._index}",
                            requires_grad=False)
        cnt = Parameter(Tensor(0, mstype.int32),
                        name=f"{self._suffix}_cnt_{self._index}",
                        requires_grad=False)
        ASDBase._index += 1
        return pre_val, min_val, max_val, cnt


class RmsNormASD(ASDBase):
    """
    RmsNorm with ASD.
"""

    def __init__(self, *args, **kwargs):
        super().__init__(OriginRmsNorm, *args, **kwargs)
        self.pre_val, self.min_val, self.max_val, self.cnt = self.generate_params()

    def __call__(self, input_x, gamma):
        if self.enable_check:
            input_x = self.check_op(
                input_x, self.pre_val, self.min_val, self.max_val, self.cnt, None)
            self.cnt += 1
        return self.op(input_x, gamma)


class LayerNormASD(ASDBase):
    """
    LayerNorm with ASD.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(OriginLayerNorm, *args, **kwargs)
        self.pre_val, self.min_val, self.max_val, self.cnt = self.generate_params()

    def __call__(self, input_x, gamma, beta):
        if self.enable_check:
            input_x = self.check_op(
                input_x, self.pre_val, self.min_val, self.max_val, self.cnt, None)
            self.cnt += 1
        return self.op(input_x, gamma, beta)


def _silent_check():
    if os.environ.get(NPU_ASD_ENABLE) == "1":
        operations.LayerNorm = LayerNormASD
        operations.RmsNorm = RmsNormASD
