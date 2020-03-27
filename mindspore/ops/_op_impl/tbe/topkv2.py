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

"""TopKV2 op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "TopKV2",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "top_k_v2.so",
    "compute_cost": 10,
    "kernel_name": "top_k_v2",
    "partial_flag": true,
    "attr": [
        {
            "name": "k",
            "param_type": "required",
            "type": "int",
            "value": "all"
        },
        {
            "name": "sorted",
            "param_type": "required",
            "type": "bool",
            "value": "all"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float16"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "input_indices",
            "need_compile": false,
            "param_type": "optional",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "values",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "int32"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "indices",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _topk_v2_tbe():
    """TopKV2 TBE register"""
    return
