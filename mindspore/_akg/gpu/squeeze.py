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

"""squeeze"""
import _akg.topi as topi
import _akg.tvm as tvm

def Squeeze(x, axis=None):
    """
    Remove the dimensions which have shape size 1.

    Args:
        x (tvm.tensor.Tensor): Tensor, input whose shape is to be squeeze.
        axis (Union[list, tuple, int, None]): specify which size 1 dimension to be removed.

    Returns:
        tvm.tensor.Tensor, has the same type and element as x, but some size 1 dimensions are removed.
    """
    return topi.squeeze(x, axis)


def gpu_schedule_Squeeze(outs):
    """
    gpu schedule Squeeze.

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
        sch = topi.cuda.schedule_injective(outs)
    return sch
