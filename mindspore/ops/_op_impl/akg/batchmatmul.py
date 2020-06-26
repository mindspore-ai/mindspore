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
    "imply_type": "AutoDiff",
    "fusion_type": "OPAQUE",
    "attr": [
        {
            "name": "transpose_a",
            "param_type": "optional",
            "type": "bool"
        },
        {
            "name": "transpose_b",
            "param_type": "optional",
            "type": "bool"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16"
            ],
            "format": [
                "FRACTAL_NZ"
            ],
            "name": "x1"
        },
        {
            "index": 1,
            "dtype": [
                "float16"
            ],
            "format": [
                "FRACTAL_NZ"
            ],
            "name": "x2"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16"
            ],
            "format": [
                "FRACTAL_NZ"
            ],
            "name": "output"
        }
    ]
}""")
def _batchmatmul_akg():
    """BatchMatMul AKG register"""
    return
