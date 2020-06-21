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

"""FusedBatchNorm op"""

from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "FusedBatchNorm",
    "imply_type": "AutoDiff",
    "fusion_type": "OPAQUE",
    "attr": [
        {
            "name": "momentum",
            "param_type": "optional",
            "type": "float"
        },
        {
            "name": "epsilon",
            "param_type": "optional",
            "type": "float"
        },
        {
            "name": "data_format",
            "param_type": "optional",
            "type": "listStr"
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
            "name": "x"
        },
        {
            "index": 1,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "scale"
        },
        {
            "index": 2,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "b"
        },
        {
            "index": 3,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "mean"
        },
        {
            "index": 4,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "variance"
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
            "name": "y"
        },
        {
            "index": 1,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "running_mean"
        },
        {
            "index": 2,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "running_variance"
        },
        {
            "index": 3,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "save_mean"
        },
        {
            "index": 4,
            "dtype": [
                "float32"
            ],
            "format": [
                "NC1HWC0"
            ],
            "name": "save_inv_variance"
        }
    ]
}""")
def _fused_batch_norm_akg():
    """FusedBatchNorm AutoDiff register"""
    return
