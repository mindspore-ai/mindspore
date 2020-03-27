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

"""ConfusionTransposeD op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "ConfusionTransposeD",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "confusion_transpose_d.so",
    "compute_cost": 10,
    "kernel_name": "confusion_transpose_d",
    "partial_flag": true,
    "attr":[
        {
            "name":"perm",
            "param_type":"required",
            "type":"listInt",
            "value":"all"
        },
        {
            "name":"shape",
            "param_type":"required",
            "type":"listInt",
            "value":"all"
        },
        {
            "name":"transpose_first",
            "param_type":"required",
            "type":"bool",
            "value":"all"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                "uint64", "float16", "float", "int8", "int16", "int32", "int64", "uint8", "uint16",
                "uint32", "uint64"
            ],
            "format": [
               "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ",
               "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "DefaultFormat", "DefaultFormat",
               "DefaultFormat", "DefaultFormat", "DefaultFormat", "DefaultFormat", "DefaultFormat",
               "DefaultFormat", "DefaultFormat", "DefaultFormat"
            ],
            "name": "x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                "uint64", "float16", "float", "int8", "int16", "int32", "int64", "uint8", "uint16",
                "uint32", "uint64"
            ],
            "format": [
               "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ",
               "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "DefaultFormat", "DefaultFormat",
               "DefaultFormat", "DefaultFormat", "DefaultFormat", "DefaultFormat", "DefaultFormat",
               "DefaultFormat", "DefaultFormat", "DefaultFormat"
            ],
            "name": "y",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _confusion_transpose_d_tbe():
    """ConfusionTransposeD TBE register"""
    return
