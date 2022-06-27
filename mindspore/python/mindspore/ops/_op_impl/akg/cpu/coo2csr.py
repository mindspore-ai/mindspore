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

"""COO2CSR op"""
from mindspore.ops.op_info_register import op_info_register, AkgCpuRegOp, DataType

coo2csr_op_info = AkgCpuRegOp("COO2CSR") \
    .fusion_type("OPAQUE") \
    .input(0, "row_indices") \
    .output(0, "output") \
    .dtype_format(DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .get_op_info()

@op_info_register(coo2csr_op_info)
def _coo2csr_akg():
    """COO2CSR AutoDiff register"""
    return
