# Copyright 2022 Huawei Technologies Co., Ltd
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
"""ascend custom op: add by tik"""
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check
from tbe import tik


@register_op_compute("AddTik")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def add_tik(x1, x2, y, kernel_name="add_tik"):
    """add dsl impl function"""
    tik_instance = tik.Tik()
    x1_shape = x1.get("shape")
    x2_shape = x2.get("shape")
    y_shape = y.get("shape")

    data_a = tik_instance.Tensor(
        "float16", x1_shape, name="x1", scope=tik.scope_gm)
    data_b = tik_instance.Tensor(
        "float16", x2_shape, name="x2", scope=tik.scope_gm)
    data_c = tik_instance.Tensor(
        "float16", y_shape, name="y", scope=tik.scope_gm)
    data_a_ub = tik_instance.Tensor(
        "float16", x1_shape, name="data_A_ub", scope=tik.scope_ubuf)
    data_b_ub = tik_instance.Tensor(
        "float16", x2_shape, name="data_B_ub", scope=tik.scope_ubuf)
    data_c_ub = tik_instance.Tensor(
        "float16", y_shape, name="data_C_ub", scope=tik.scope_ubuf)

    tik_instance.data_move(data_a_ub, data_a, 0, 1, 128 // 16, 0, 0)
    tik_instance.data_move(data_b_ub, data_b, 0, 1, 128 // 16, 0, 0)
    tik_instance.vec_add(
        128, data_c_ub[0], data_a_ub[0], data_b_ub[0], 1, 8, 8, 8)
    tik_instance.data_move(data_c, data_c_ub, 0, 1, 128 // 16, 0, 0)
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[data_a, data_b], outputs=[data_c])

    return tik_instance
