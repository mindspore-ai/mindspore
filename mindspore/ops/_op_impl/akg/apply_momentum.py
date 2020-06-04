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

"""ApplyMomentum op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "ApplyMomentum",
    "imply_type": "AutoDiff",
    "fusion_type": "ELEMWISE",
    "attr": [
        {
            "name": "use_nesterov",
            "param_type": "optional",
            "type": "bool"
        },
        {
            "name": "gradient_scale",
            "param_type": "optional",
            "type": "float"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float32","float32","float32"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","FracZ"
            ],
            "name": "variable"
        },
        {
            "index": 1,
            "dtype": [
                "float32","float32","float32"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","FracZ"
            ],
            "name": "accumulation"
        },
        {
            "index": 2,
            "dtype": [
                "float32","float32","float32"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","DefaultFormat"
            ],
            "name": "learning_rate"
        },
        {
            "index": 3,
            "dtype": [
                "float32","float32","float32"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","FracZ"
            ],
            "name": "gradient"
        },
        {
            "index": 4,
            "dtype": [
                "float32","float32","float32"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","DefaultFormat"
            ],
            "name": "momentum"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float32","float32","float32"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","FracZ"
            ],
            "name": "output"
        }
    ]
}""")
def _apply_momentum_akg():
    """ApplyMomentum AutoDiff register"""
    return
