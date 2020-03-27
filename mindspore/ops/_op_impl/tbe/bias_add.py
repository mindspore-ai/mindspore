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

"""BiasAdd op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "BiasAdd",
    "imply_type": "TBE",
    "fusion_type": "COMMREDUCE",
    "async_flag": false,
    "binfile_name": "bias_add.so",
    "compute_cost": 10,
    "kernel_name": "bias_add",
    "partial_flag": true,
    "attr": [
        {
            "name": "data_format",
            "param_type": "required",
            "type": "str",
            "value": "all"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "int32", "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat", "DefaultFormat"
            ],
            "name": "x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "int32", "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat", "DefaultFormat"
            ],
            "name": "bias",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "int32", "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat", "DefaultFormat"
            ],
            "name": "y",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _bias_add_tbe():
    """BiasAdd TBE register"""
    return
