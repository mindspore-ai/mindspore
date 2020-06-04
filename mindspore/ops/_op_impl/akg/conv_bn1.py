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

"""ConvBN1 op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "ConvBN1",
    "imply_type": "AutoDiff",
    "fusion_type": "CONVLUTION",
    "attr": [
        {
            "name": "x_shape",
            "param_type": "required",
            "type": "listInt"
        },
        {
            "name": "w_shape",
            "param_type": "required",
            "type": "listInt"
        },
        {
            "name": "pad_list",
            "param_type": "required",
            "type": "listInt"
        },
        {
            "name": "stride",
            "param_type": "optional",
            "type": "int"
        },
        {
            "name": "dilation",
            "param_type": "optional",
            "type": "int"
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
            "name": "x"
        },
        {
            "index": 1,
            "dtype": [
                "float16"
            ],
            "format": [
                "FracZ"
            ],
            "name": "w"
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
            "name": "conv_res_16"
        },
        {
            "index": 1,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "var_part"
        },
        {
            "index": 2,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "mean"
        }
    ]
}""")
def _conv_bn1_akg():
    """ConvBN1 AutoDiff register"""
    return
