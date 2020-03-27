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

"""cast"""
import logging
import akg.tvm
from akg.ops.math import cast
from akg.topi.generic import schedule_elemwise

def Cast(x, dst_type):
    """cast."""
    return cast.cast(x, dst_type)


def gpu_schedule_Cast(outs):
    """
    gpu schedule for cast.

    Args:
        outs (tvm.tensor.Tensor): outputs of compute.

    Returns:
        sch (schedule.Schedule): The created schedule.
    """
    device = 'cuda'
    ctx = akg.tvm.context(device, 0)
    if not ctx.exist:
        logging.info("Skip because %s is not enabled", device)
        return None
    with akg.tvm.target.create(device):
        sch = schedule_elemwise(outs)
    return sch
