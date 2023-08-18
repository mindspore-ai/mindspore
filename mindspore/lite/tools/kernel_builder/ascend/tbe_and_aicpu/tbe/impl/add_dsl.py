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
"""ascend custom op: add by dsl"""
import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check


@register_op_compute("add_dsl")
def add_dsl_compute(x1, x2, y, kernel_name="add_dsl"):
    res = tbe.vadd(x1, x2)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def add_dsl(x1, x2, y, kernel_name="add_dsl"):
    """add dsl impl function"""
    data_x1 = tvm.placeholder(
        x1.get("shape"), dtype=x1.get("dtype"), name="data_x1")
    data_x2 = tvm.placeholder(
        x2.get("shape"), dtype=x2.get("dtype"), name="data_x2")

    res = add_dsl_compute(data_x1, data_x2, y, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x1, data_x2, res]}
    tbe.build(schedule, config)
