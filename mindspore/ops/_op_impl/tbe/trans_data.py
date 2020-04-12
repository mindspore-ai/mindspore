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

"""TransData op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "TransData",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "trans_data.so",
    "compute_cost": 10,
    "kernel_name": "trans_data",
    "partial_flag": true,
    "attr": [
        {
            "name": "src_format",
            "param_type": "required",
            "type": "str",
            "value": "DefaultFormat,NC1HWC0,FracZ,FRACTAL_NZ,HWCN,C1HWNCoC0"
        },
        {
            "name": "dst_format",
            "param_type": "required",
            "type": "str",
            "value": "DefaultFormat,NC1HWC0,FracZ,FRACTAL_NZ,HWCN,C1HWNCoC0"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "bool",
                "float","float","float","float","float","float","float","float","float","float",
                "float16","float16","float16","float16","float16","float16","float16","float16","float16","float16",
                "uint16","uint16","uint16","uint16","uint16","uint16","uint16","uint16","uint16","uint16"
            ],
            "format": [
                "DefaultFormat",
                "DefaultFormat","DefaultFormat","DefaultFormat","FracZ","FRACTAL_NZ","NC1HWC0","HWCN","HWCN","C1HWNCoC0","FracZ",
                "DefaultFormat","DefaultFormat","DefaultFormat","FracZ","FRACTAL_NZ","NC1HWC0","HWCN","HWCN","C1HWNCoC0","FracZ",
                "DefaultFormat","DefaultFormat","DefaultFormat","FracZ","FRACTAL_NZ","NC1HWC0","HWCN","HWCN","C1HWNCoC0","FracZ"
            ],
            "name": "src",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "bool",
                "float","float","float","float","float","float","float","float","float","float",
                "float16","float16","float16","float16","float16","float16","float16","float16","float16","float16",
                "uint16","uint16","uint16","uint16","uint16","uint16","uint16","uint16","uint16","uint16"
            ],
            "format": [
                "NC1HWC0",
                "NC1HWC0","FRACTAL_NZ","FracZ","DefaultFormat","DefaultFormat","DefaultFormat","FracZ","C1HWNCoC0","HWCN","HWCN",
                "NC1HWC0","FRACTAL_NZ","FracZ","DefaultFormat","DefaultFormat","DefaultFormat","FracZ","C1HWNCoC0","HWCN","HWCN",
                "NC1HWC0","FRACTAL_NZ","FracZ","DefaultFormat","DefaultFormat","DefaultFormat","FracZ","C1HWNCoC0","HWCN","HWCN"
            ],
            "name": "dst",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _trans_data_tbe():
    """TransData TBE register"""
    return
