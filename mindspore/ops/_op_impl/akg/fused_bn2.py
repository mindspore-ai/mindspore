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

"""FusedBN2 op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "FusedBN2",
    "imply_type": "AutoDiff",
    "fusion_type": "COMMREDUCE",
    "attr": [
        {
            "name": "momentum",
            "param_type": "optional",
            "type": "float"
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
            "name": "mean"
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
            "name": "running_mean"
        },
        {
            "index": 3,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "running_var"
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
        }
    ]
}""")
def _fused_bn2_akg():
    """FusedBN2 AutoDiff register"""
    return
