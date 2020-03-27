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

"""mean op compute and schedule"""
import akg.tvm as tvm
from akg.ops.math.mean import mean
from .default_schedule import DEFAULT_GPU_THREAD

def Mean(x, axis=None, keepdims=True):
    """mean."""
    outs = mean(x, axis, keepdims)

    # remove useless mean_output
    if isinstance(outs, tuple):
        outs = outs[0]
    if outs.op.name == "mean_output":
        outs = outs.op.input_tensors[0]
    return outs


def gpu_schedule_Mean(outs):
    """
    gpu schedule function for mean.

    Args:
        outs (tvm.tensor.Tensor): outputs of compute.

    Returns:
        sch (schedule.Schedule): The created schedule.
    """
    out = outs[0] if isinstance(outs, list) else outs

    device = "cuda"
    with tvm.target.create(device):
        sch = tvm.create_schedule(out.op)
        if out.op.name == "T_divide":
            tensor_c = out
        else:  # squeeze
            tensor_c = out.op.input_tensors[0]

        tensor_b = tensor_c.op.input_tensors[0]
        if len(tensor_c.op.axis) >= 2:
            sch[tensor_b].compute_at(sch[tensor_c], tensor_c.op.axis[1])
        else:
            sch[tensor_b].compute_at(sch[tensor_c], tensor_c.op.axis[0])

        bx, tx = sch[tensor_c].split(tensor_c.op.axis[0], factor=DEFAULT_GPU_THREAD)
        sch[tensor_c].bind(bx, tvm.thread_axis("blockIdx.x"))
        sch[tensor_c].bind(tx, tvm.thread_axis("threadIdx.x"))
    return sch

def SimpleMean(x):
    """
    SimpleMean compute the mean of the input 4D Tensor over last two axises and keep reduced dimensions.

    Args:
        x (tvm.tensor.Tensor): Tensor of type float16, float32.

    Returns:
        tvm.tensor.Tensor, has the same type as x, output shape will be (a, b, 1, 1) if input Tensor x is (a, b, c, d).
    """
    axis = (2, 3)
    keepdims = True
    return Mean(x, axis, keepdims)


def gpu_schedule_SimpleMean(outs):
    """gpu schedule function for SimpleMean."""
    return gpu_schedule_Mean(outs)
