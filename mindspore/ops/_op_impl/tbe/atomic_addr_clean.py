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

"""AtomicAddrClean op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "AtomicAddrClean",
    "imply_type": "TBE",
    "fusion_type": "ELEMWISE",
    "async_flag": false,
    "binfile_name": "atomic_addr_clean.so",
    "compute_cost": 10,
    "kernel_name": "atomic_addr_clean",
    "partial_flag": true,
    "attr": [
        {
            "name": "automic_add_mem_size",
            "param_type": "required",
            "type": "listInt",
            "value": "all"
        }
    ],
    "inputs": [
    ],
    "outputs": [
    ]
}""")
def _atomic_addr_clean_tbe():
    """AtomicAddrClean TBE register"""
    return
