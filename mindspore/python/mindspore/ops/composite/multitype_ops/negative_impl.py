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

"""Implementation for internal polymorphism `negative` operations."""

from __future__ import absolute_import
from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.common import CSRTensor, COOTensor


negative = base.MultitypeFuncGraph("negative", True)
"""
`negative` is a metafuncgraph object which will give the negative of an object according to its input type
using ".register" decorator.
"""
negative.set_need_raise()


@negative.register("Number")
def _neg_scalar(x):
    """
    Returns the negative value of scalar x.

    Outputs:
       Number, negative value of x.
   """
    return F.scalar_usub(x)


@negative.register("Tensor")
def _negative_tensor(x):
    """
    Returns the negative value of tensor x by element-wise.

    Returns:
       Tensor, negative value of x by element-wise.
   """
    return F.neg_tensor(x)


@negative.register("CSRTensor")
def _negative_csrtensor(x):
    """
    Returns the negative value of tensor x by element-wise.

    Returns:
       CSRTensor, negative value of x by element-wise.
   """
    return CSRTensor(x.indptr, x.indices, F.neg_tensor(x.values), x.shape)


@negative.register("COOTensor")
def _negative_cootensor(x):
    """
    Returns the negative value of tensor x by element-wise.

    Returns:
       COOTensor, negative value of x by element-wise.
   """
    return COOTensor(x.indices, F.neg_tensor(x.values), x.shape)
