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

"""Conv2DBackpropFilter op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "Conv2DBackpropFilter",
    "imply_type": "AutoDiff",
    "fusion_type": "CONVLUTION",
    "attr": [
        {
            "name": "input_shape",
            "param_type": "required",
            "type": "listInt"
        },
        {
            "name": "filter_sizes",
            "param_type": "required",
            "type": "listInt"
        },
        {
            "name": "stride",
            "param_type": "optional",
            "type": "int"
        },
        {
            "name": "pad_list",
            "param_type": "required",
            "type": "listInt"
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
            "name": "out_backprop"
        },
        {
            "index": 1,
            "dtype": [
                "float16"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "input"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float32"
            ],
            "format": [
                "FracZ"
            ],
            "name": "output"
        }
    ]
}""")
def _conv2d_backprop_filter_akg():
    """Conv2DBackpropFilter AutoDiff register"""
    return
