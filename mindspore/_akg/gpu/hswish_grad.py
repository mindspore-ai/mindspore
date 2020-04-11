# Copyright 2019 Huawei Technologies Co., Ltd
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

"""HswishGrad"""
import _akg.topi as topi
import _akg.tvm as tvm


def HswishGrad(y_grad, x):
    """
    HswishGrad
    Args:
        y_grad:
        x:

    Returns:

    """
    shape = x.shape

    res0 = tvm.compute(shape, lambda *i: tvm.if_then_else(x(*i) <= -3, 0, y_grad(*i) * (2 * x(*i) + 3) / 6))
    res6 = tvm.compute(shape, lambda *i: tvm.if_then_else(x(*i) >= 3, y_grad(*i), res0(*i)))
    return res6


def gpu_schedule_HswishGrad(outs):
    """
    gpu schedule HswishGrad
    Args:
        outs:

    Returns:

    """
    device = 'cuda'
    ctx = tvm.context(device, 0)
    if not ctx.exist:
        raise SystemError("Skip because %s is not enabled" % device)

    with tvm.target.create(device):
        sch = topi.cuda.schedule_elemwise(outs)
    return sch
