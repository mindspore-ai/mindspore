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
"""less_equal"""
import _akg.tvm
from _akg.ops.math import less_equal
from _akg.topi.generic import schedule_elemwise

def LessEqual(x, y):
    """LessEqual."""
    return less_equal.less_equal(x, y)


def gpu_schedule_LessEqual(outs):
    """
    GPU schedule for LessEqual.

    Args:
        outs (tvm.tensor.Tensor): Outputs of compute.

    Returns:
        sch (schedule.Schedule): The created schedule.
    """
    device = 'cuda'
    ctx = _akg.tvm.context(device, 0)
    if not ctx.exist:
        raise SystemError("Skip because %s is not enabled" % device)
    with _akg.tvm.target.create(device):
        sch = schedule_elemwise(outs)
    return sch
