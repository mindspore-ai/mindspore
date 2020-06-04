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

"""BNGrad3 op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "BNGrad3",
    "imply_type": "AutoDiff",
    "fusion_type": "ELEMWISE",
    "attr": [

    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float32"
            ],
            "format": [
                "NC1HWC0", "NC1HWC0"
            ],
            "name": "dy"
        },
        {
            "index": 1,
            "dtype": [
                "float32", "float32"
            ],
            "format": [
                "NC1HWC0", "NC1HWC0"
            ],
            "name": "rs"
        },{
            "index": 2,
            "dtype": [
                "float32", "float32"
            ],
            "format": [
                "NC1HWC0", "NC1HWC0"
            ],
            "name": "dgamma_dx"
        },
        {
            "index": 3,
            "dtype": [
                "float32", "float32"
            ],
            "format": [
                "NC1HWC0", "NC1HWC0"
            ],
            "name": "dbeta_dx"
        },
        {
            "index": 4,
            "dtype": [
                "float32", "float32"
            ],
            "format": [
                "NC1HWC0", "NC1HWC0"
            ],
            "name": "data_minus_mean"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float32"
            ],
            "format": [
                "NC1HWC0", "NC1HWC0"
            ],
            "name": "output"
        }
    ]
}""")
def _bn3_grad_akg():
    """BNGrad3 AutoDiff register"""
    return
