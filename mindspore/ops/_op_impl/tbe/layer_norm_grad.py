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

"""LayerNormGrad op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "LayerNormGrad",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "layer_norm_grad.so",
    "compute_cost": 10,
    "kernel_name": "layer_norm_grad",
    "partial_flag": true,
    "attr": [

    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float","float"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0"
            ],
            "name": "dy",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float16","float16","float","float"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0"
            ],
            "name": "x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 2,
            "dtype": [
                "float16","float16","float","float"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0"
            ],
            "name": "variance",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 3,
            "dtype": [
                "float16","float16","float","float"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0"
            ],
            "name": "mean",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 4,
            "dtype": [
                "float16","float16","float","float"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0"
            ],
            "name": "gamma",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
         {
            "index": 0,
            "dtype": [
                "float16","float16","float","float"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0"
            ],
            "name": "pd_x",
            "param_type": "required"
        },
        {
            "index": 1,
            "dtype": [
                "float16","float16","float","float"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0"
            ],
            "name": "pd_gamma",
            "param_type": "required"
        },
        {
            "index": 2,
            "dtype": [
                "float16","float16","float","float"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0"
            ],
            "name": "pd_beta",
            "param_type": "required"
        }
    ]
}""")
def _layer_norm_grad_tbe():
    """LayerNormGrad TBE register"""
    return
