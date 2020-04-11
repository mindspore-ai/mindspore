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

"""relu6"""
import _akg.topi as topi
import _akg.tvm as tvm
from _akg.topi import tag

@tvm.tag_scope(tag=tag.ELEMWISE)
def topi_nn_relu6(x):
    """topi nn relu6."""
    return tvm.compute(x.shape, lambda *i: tvm.min(tvm.max(x(*i), tvm.const(0, x.dtype)), tvm.const(6, x.dtype)))

def ReLU6(x):
    """
    Compute elementwise with function: min(max(x, 0), 6).

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has same type and shape as input.
    """
    return topi_nn_relu6(x)


def gpu_schedule_ReLU6(outs):
    """
    gpu schedule ReLU6.

    Args:
        outs (tvm.tensor.Tensor): outputs of compute.

    Returns:
        sch (schedule.Schedule): The created schedule.
    """
    device = 'cuda'
    ctx = tvm.context(device, 0)
    if not ctx.exist:
        raise SystemError("Skip because %s is not enabled" % device)
    with tvm.target.create(device):
        sch = topi.cuda.schedule_elemwise(outs)
    return sch
