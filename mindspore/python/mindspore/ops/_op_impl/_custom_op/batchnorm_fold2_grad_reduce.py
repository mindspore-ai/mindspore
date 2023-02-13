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

"""_BatchNormFold2GradReduce op"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.common.buildcfg import build_config
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

SHAPE_SIZE_LIMIT = 2147483648

batchnorm_fold2_grad_reduce_op_info = TBERegOp("BatchNormFold2GradReduce") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("batchnorm_fold2_grad_reduce.so") \
    .compute_cost(10) \
    .kernel_name("batchnorm_fold2_grad_reduce") \
    .partial_flag(True) \
    .input(0, "dout", None, "required", None) \
    .input(1, "x", None, "required", None) \
    .output(0, "dout_reduce", True, "required", "all") \
    .output(1, "dout_x_reduce", True, "required", "all") \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(batchnorm_fold2_grad_reduce_op_info)
def _batchnorm_fold2_grad_reduce_tbe():
    """_BatchNormFold2GradReduce TBE register"""
    return


@fusion_manager.register("batchnorm_fold2_grad_reduce")
def batchnorm_fold2_grad_reduce_compute(dout, x, dout_args, kernel_name="batchnorm_fold2_grad_reduce"):
    """_BatchNormFold2GradReduce compute"""
    dtype = dout_args.get("dtype")
    dout_format = dout_args.get("format")
    ori_format = dout_args.get("ori_format")
    shape = dout_args.get("shape")

    if dtype == "float16":
        dout = te.lang.cce.cast_to(dout, "float32")
        x = te.lang.cce.cast_to(x, "float32")

    dout_x = te.lang.cce.vmul(dout, x)
    if dout_format == "NC1HWC0":
        axis = [0, 2, 3]
        dout_reduce, dout_x_reduce = te.lang.cce.tuple_sum([dout, dout_x], axis, True)
    else:
        axis = list(range(len(shape)))
        if ori_format == "NCHW":
            axis.pop(1)
        for _, i in enumerate(range(len(shape))):
            if shape[i] == 1 and i in axis:
                axis.remove(i)
        dout_reduce = te.lang.cce.sum(dout, axis, False)
        dout_x_reduce = te.lang.cce.sum(dout_x, axis, False)
    return [dout_reduce, dout_x_reduce]


@util.check_input_type(dict, dict, dict, dict, str)
def batchnorm_fold2_grad_reduce(dout, x, dout_reduce, dout_x_reduce, kernel_name="batchnorm_fold2_grad_reduce"):
    """_BatchNormFold2GradReduce op"""
    shape = x.get("shape")
    x_format = x.get("format")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    inp_dtype = x.get("dtype").lower()
    if inp_dtype not in ["float16", "float32"]:
        raise RuntimeError("Dtype of input only support float16, float32")
    dout_t = tvm.placeholder(shape, name="dout", dtype=inp_dtype)
    x_t = tvm.placeholder(shape, name="x", dtype=inp_dtype)

    res_list = batchnorm_fold2_grad_reduce_compute(dout_t, x_t, dout, kernel_name)

    if x_format == "NC1HWC0":
        with tvm.target.cce():
            sch = generic.auto_schedule(res_list)
        tensor_list = [dout_t, x_t] + list(res_list)
        config = {"print_ir": False,
                  "name": kernel_name,
                  "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)
        return
    from impl.bn_training_reduce import bn_training_reduce_schedule_nd
    sch, tensor_list = bn_training_reduce_schedule_nd(res_list)
    with build_config():
        tvm.build(sch, tensor_list, "cce", name=kernel_name)
