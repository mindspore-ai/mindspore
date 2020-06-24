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

"""BNGrad1 op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "BNGrad2",
    "imply_type": "AutoDiff",
    "fusion_type": "COMMREDUCE",
    "attr": [
        {
            "name": "eps",
            "param_type": "optional",
            "type": "float"
        },
        {
            "name": "data_shape",
            "param_type": "optional",
            "type": "listInt"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "dgamma_red_hw"
        },
        {
            "index": 1,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "dbeta_red_hw"
        },{
            "index": 2,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "variance"
        },
        {
            "index": 3,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "gamma"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "output"
        },
        {
            "index": 1,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "output"
        },
        {
            "index": 2,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "output"
        },
        {
            "index": 3,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "output"
        },
        {
            "index": 4,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "output"
        }
    ]
}""")
def _bn2_grad_akg():
    """BNGrad2 AutoDiff register"""
    return
