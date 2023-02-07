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
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util

from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

@fusion_manager.register("square")
def square_compute(input_x, output_y):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is square

    Returns
    -------
    res : tvm.tensor
        the result of square
    """
    res = te.lang.cce.vmul(input_x, input_x)
    return res


cus_square_op_info = TBERegOp("CusSquare") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("square.so") \
    .compute_cost(10) \
    .kernel_name("CusSquareImpl") \
    .partial_flag(True) \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(cus_square_op_info)
def CusSquareImpl(input_x, output_y, kernel_name="CusSquareImpl"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "square"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()

    shape = util.shape_refine(shape)
    data = tvm.placeholder(shape, name="data", dtype=dtype.lower())

    with tvm.target.cce():
        res = square_compute(data, output_y)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)
