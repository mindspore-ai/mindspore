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

"""default schedule function for GPU"""
from queue import Queue

import akg.tvm as tvm

DEFAULT_GPU_THREAD = 1024


def default_schedule(outs):
    """
    default schedule function.

    Args:
        outs (Union[tvm.tensor.Tensor, list[tvm.tensor.Tensor]]): outputs of compute.

    Returns:
        sch (schedule.Schedule): The created schedule.
    """
    if not isinstance(outs, tvm.tensor.Tensor) and not isinstance(outs, list):
        raise ValueError("outs should be list of akg.tvm.tensor.Tensor or akg.tvm.tensor.Tensor")
    device = 'cuda'
    ctx = tvm.context(device, 0)
    if not ctx.exist:
        raise SystemError("Skip because %s is not enabled" % device)
    outs_list = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    with tvm.target.create(device):
        sch = tvm.create_schedule(outs_list[0].op)
        outputs_tensor = Queue()
        outputs_tensor.put(outs_list[0])
        op_list = []
        while not outputs_tensor.empty():
            out = outputs_tensor.get()
            if out.op not in op_list and isinstance(out.op, tvm.tensor.ComputeOp):
                op_list.append(out.op)
                for input_tensor in out.op.input_tensors:
                    outputs_tensor.put(input_tensor)
        for op in op_list:
            stage = sch[op.output(0)]
            bx, tx = stage.split(op.axis[0], factor=DEFAULT_GPU_THREAD)
            stage.bind(bx, tvm.thread_axis("blockIdx.x"))
            stage.bind(tx, tvm.thread_axis("threadIdx.x"))
    return sch
