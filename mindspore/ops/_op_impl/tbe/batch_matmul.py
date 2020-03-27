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

"""BatchMatMul op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "BatchMatMul",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "batch_matmul.so",
    "compute_cost": 10,
    "kernel_name": "batch_matmul",
    "partial_flag": true,
    "attr": [
        {
            "name": "transpose_x1",
            "param_type": "required",
            "type": "bool",
            "value": "all"
        },
        {
            "name": "transpose_x2",
            "param_type": "required",
            "type": "bool",
            "value": "all"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float16","float","float","int32","int32"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","FRACTAL_NZ","DefaultFormat","DefaultFormat","DefaultFormat","DefaultFormat"
            ],
            "name": "x1",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float16","float16","float16","float","float","int32","int32"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","FRACTAL_NZ","DefaultFormat","DefaultFormat","DefaultFormat","DefaultFormat"
            ],
            "name": "x2",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 2,
            "dtype": [
                "float16","float16","float16","float","float","int32","int32"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","DefaultFormat","DefaultFormat","DefaultFormat","DefaultFormat","DefaultFormat"
            ],
            "name": "bias",
            "need_compile": false,
            "param_type": "optional",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float16","float","float","int32","int32"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","FRACTAL_NZ","DefaultFormat","DefaultFormat","DefaultFormat","DefaultFormat"
            ],
            "name": "y",
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _batch_matmul_tbe():
    """BatchMatMul TBE register"""
    return
