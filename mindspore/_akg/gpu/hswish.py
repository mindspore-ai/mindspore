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

"""hswish"""
import _akg.topi as topi
import _akg.tvm as tvm
from _akg.topi import tag


@tvm.tag_scope(tag=tag.ELEMWISE)
def topi_nn_hswish(x):
    """
    topi hswish
    Args:
        x:

    Returns:

    """
    return tvm.compute(x.shape, lambda *i: tvm.if_then_else(x(*i) <= -3, 0,
                                                            tvm.if_then_else(x(*i) >= 3, x(*i),
                                                                             x(*i) * (x(*i) + 3) / 6)))


def Hswish(x):
    """
    Hswish
    Args:
        x:

    Returns:

    """
    return topi_nn_hswish(x)


def gpu_schedule_Hswish(outs):
    """
    gpu schedule Hswish
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
