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

"""mean_grad"""
import akg.tvm as tvm
import akg
from akg.ops.math import mean
from .default_schedule import DEFAULT_GPU_THREAD


def mean_ad(head, input_shape, axis, keepdims):
    """mean autodiff."""
    tensor_a = tvm.placeholder(input_shape, head.dtype, "A")
    tensor_b = mean.mean(tensor_a, axis, keepdims)

    # remove useless mean_output
    if isinstance(tensor_b, tuple):
        tensor_b = tensor_b[0]
    if tensor_b.op.name == "mean_output":
        tensor_b = tensor_b.op.input_tensors[0]

    jacs = list(akg.differentiate(tensor_b, [tensor_a], head))
    return jacs[0]


def MeanGrad(y_grad, input_shape, axis=None, keepdims=True):
    """Mean Grad."""
    if axis is None and not keepdims:
        raise ValueError("Mean not support (axis=None && keepdims=False)  now")
    return mean_ad(y_grad, input_shape, axis, keepdims)


def gpu_schedule_MeanGrad(outs):
    """gpu schedule MeanGrad."""
    out = outs[0] if isinstance(outs, list) else outs

    device = "cuda"
    with tvm.target.create(device):
        sch = tvm.create_schedule(out.op)
        tensor_c = out
        tensor_b = tensor_c.op.input_tensors[0]
        if len(tensor_c.op.axis) >= 2:
            sch[tensor_b].compute_at(sch[tensor_c], tensor_c.op.axis[1])
        else:
            sch[tensor_b].compute_at(sch[tensor_c], tensor_c.op.axis[0])

        bx, tx = sch[tensor_c].split(tensor_c.op.axis[0], factor=DEFAULT_GPU_THREAD)
        sch[tensor_c].bind(bx, tvm.thread_axis("blockIdx.x"))
        sch[tensor_c].bind(tx, tvm.thread_axis("threadIdx.x"))

    return sch

def SimpleMeanGrad(HEAD, input_shape):
    """
    Compute Simple Mean Grad.

    Args:
        HEAD (tvm.tensor.Tensor): output gradient, dy, defined in Primitive.
        input_shape (Union[list[int], tuple[int]]): shape of mean input, x.shape.

    Returns:
        tvm.tensor.Tensor, gradient of mean input.
    """
    axis = (2, 3)
    keepdims = True
    return MeanGrad(HEAD, input_shape, axis, keepdims)


def gpu_schedule_SimpleMeanGrad(outs):
    """
    gpu schedule SimpleMeanGrad.

    Args:
        outs (tvm.tensor.Tensor): outputs of compute.

    Returns:
        sch (schedule.Schedule): The created schedule.
    """
    return gpu_schedule_MeanGrad(outs)
