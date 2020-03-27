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
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "apply_momentum.so",
    "compute_cost": 10,
    "kernel_name": "apply_momentum",
    "partial_flag": true,
    "attr": [
        {
            "name": "use_nesterov",
            "param_type": "optional",
            "type": "bool",
            "value": "true,false",
            "default_value":"false"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float16","float16","float","float","float","float"
            ],
            "format": [
                "NC1HWC0", "C1HWNCoC0", "DefaultFormat", "FracZ", "NC1HWC0", "DefaultFormat", "FracZ", "C1HWNCoC0"
            ],
            "name": "var",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float16","float16","float16","float16","float","float","float","float"
            ],
            "format": [
                "NC1HWC0", "C1HWNCoC0", "DefaultFormat", "FracZ", "NC1HWC0", "DefaultFormat", "FracZ", "C1HWNCoC0"
            ],
            "name": "accum",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 2,
            "dtype": [
                "float16","float16","float16","float16","float","float","float","float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat", "DefaultFormat", "DefaultFormat", "DefaultFormat", "DefaultFormat",
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "lr",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 3,
            "dtype": [
                "float16","float16","float16","float16","float","float","float","float"
            ],
            "format": [
                "NC1HWC0", "C1HWNCoC0", "DefaultFormat", "FracZ", "NC1HWC0", "DefaultFormat", "FracZ", "C1HWNCoC0"
            ],
            "name": "grad",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 4,
            "dtype": [
                "float16","float16","float16","float16","float","float","float", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat", "DefaultFormat", "DefaultFormat", "DefaultFormat", "DefaultFormat",
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "momentum",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float16","float16","float","float","float","float"
            ],
            "format": [
                "NC1HWC0", "C1HWNCoC0", "DefaultFormat", "FracZ", "NC1HWC0", "DefaultFormat", "FracZ", "C1HWNCoC0"
            ],
            "name": "var",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _apply_momentum_tbe():
    """ApplyMomentum TBE register"""
    return
