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

"""SquareSumV2 op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "SquareSumV2",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "square_sum_v2.so",
    "compute_cost": 10,
    "kernel_name": "square_sum_v2",
    "partial_flag": true,
    "attr":[
        {
            "name":"axis",
            "param_type":"optional",
            "type":"listInt",
            "value":"all"
        },
        {
            "name":"keep_dims",
            "param_type":"optional",
            "type":"bool",
            "value":"all"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float32"
            ],
            "format": [
               "DefaultFormat", "DefaultFormat"
            ],
            "name": "input_x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float32"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "output1",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float16", "float32"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "output2",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _square_sum_v2_tbe():
    """SquareSumV2 TBE register"""
    return
