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

"""Conv2D op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "Conv2D",
    "imply_type": "TBE",
    "fusion_type": "CONVLUTION",
    "async_flag": false,
    "binfile_name": "conv2d.so",
    "compute_cost": 10,
    "kernel_name": "conv2d",
    "partial_flag": true,
    "attr": [
        {
            "name": "stride",
            "param_type": "required",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "pad",
            "param_type": "required",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "dilation",
            "param_type": "required",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "offset_a",
            "param_type": "optional",
            "type": "int",
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
                "NC1HWC0"
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
                "FracZ"
            ],
            "name": "filter",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 2,
            "dtype": [
                "float16"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "bias",
            "need_compile": false,
            "param_type": "optional",
            "shape": "all"
        },
        {
            "index": 3,
            "dtype": [
                "int8"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "offset_w",
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
                "NC1HWC0"
            ],
            "name": "y",
            "need_compile": true,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _conv2d_tbe():
    """Conv2D TBE register"""
    return
