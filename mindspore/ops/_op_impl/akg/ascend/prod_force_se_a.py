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

"""ProdForceSeA op"""
from mindspore.ops.op_info_register import op_info_register, AkgAscendRegOp, DataType as DT

op_info = AkgAscendRegOp("ProdForceSeA") \
    .fusion_type("ELEMWISE") \
    .attr("natoms", "required", "int") \
    .input(0, "net_deriv_tensor") \
    .input(1, "in_deriv_tensor") \
    .input(2, "nlist_tensor") \
    .output(0, "output") \
    .dtype_format(DT.F32_Default, DT.F32_Default, DT.I32_Default, DT.F32_Default) \
    .get_op_info()


@op_info_register(op_info)
def _prod_force_se_a_akg():
    """ProdForceSeA Akg register"""
    return
