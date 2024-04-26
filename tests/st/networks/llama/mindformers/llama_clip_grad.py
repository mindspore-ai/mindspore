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
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator as validator
    from mindspore._checkparam import Rel
except ImportError:
    import mindspore._checkparam as validator
    import mindspore._checkparam as Rel


expand_dims = P.ExpandDims().add_prim_attr("grad_scale", True)
get_square_sum = C.MultitypeFuncGraph("get_square_sum")
apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@get_square_sum.register("Tensor")
def _get_square_sum(x):
    """get square summation for a Tensor."""
    norm = P.ReduceSum(False)(F.square(x), ())
    norm = expand_dims(F.cast(norm, mstype.float32), 0)
    return norm

@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, x):
    """apply global normalization for a Tensor."""
    x_dtype = F.dtype(x)
    clip_coef = clip_norm / (global_norm + 1e-6)
    clip_coef_clamped = ops.clip_by_value(clip_coef, clip_value_max=Tensor(1.0, mstype.float32),
                                          clip_value_min=Tensor(float('-inf'), mstype.float32))
    x = F.cast(x, F.dtype(clip_coef_clamped))
    x = x * clip_coef_clamped
    x = F.cast(x, x_dtype)
    return x

class ClipGradNorm(Cell):
    r"""
    Clips tensor values by the ratio of the sum of their norms.

    Args:
        max_norm (Union(float, int)): The clipping ratio. Default: 1.0
        use_norm (Union(float, None)): The global norm. Default: None

    Inputs:
        - **x** (Union(tuple[Tensor], list[Tensor])) - Input data to clip.

    Outputs:
        Tensor, a clipped Tensor.
    """

    def __init__(self, max_norm=1.0, use_norm=None):
        super(ClipGradNorm, self).__init__()
        # Add interface. This parameter is not used at present
        if use_norm is not None:
            raise ValueError(f"For '{self.cls_name}', input 'use_norm' only supports None currently, "
                             f"but got 'use_norm': {use_norm}")
        validator.check_number("clip_norm", max_norm, 0.0, Rel.GT, self.cls_name)
        self.clip_norm = Tensor([max_norm], mstype.float32)
        self.hyper_map = C.HyperMap()
        self.greater_equal = P.GreaterEqual()

    def construct(self, x):
        """
        clip gradient.

        Args:
            - **x** (Union(tuple[Tensor], list[Tensor])) - Input gradient to clip.

        Returns:
            Tensor, a clipped gradient.
        """
        square_sum = self.hyper_map(get_square_sum, x)
        global_norm = F.sqrt(F.addn(square_sum))
        cond = self.greater_equal(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        clip_x = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), x)
        return clip_x, global_norm
