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

"""Conv2DBackpropInput op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "Conv2DBackpropInput",
    "imply_type": "TBE",
    "fusion_type": "CONVLUTION",
    "async_flag": false,
    "binfile_name": "conv2d_backprop_input_d.so",
    "compute_cost": 10,
    "kernel_name": "conv2d_backprop_input_d",
    "partial_flag": true,
    "attr": [
        {
            "name": "input_sizes",
            "param_type": "required",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "stride",
            "param_type": "required",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "pad_mode",
            "param_type": "required",
            "type": "str",
            "value": "all"
        },
        {
            "name": "dilation",
            "param_type": "required",
            "type": "listInt",
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
            "name": "out_backprop",
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
def _conv2d_backprop_input_tbe():
    """Conv2DBackpropInput TBE register"""
    return
