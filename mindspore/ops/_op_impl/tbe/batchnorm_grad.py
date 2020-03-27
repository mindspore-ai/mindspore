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

"""BatchNormGrad op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "BatchNormGrad",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "batchnormgrad.so",
    "compute_cost": 10,
    "kernel_name": "batchnormgrad",
    "partial_flag": true,
    "attr": [
        {
            "name": "epsilon",
            "param_type": "optional",
            "type": "float",
            "value": "all"
        },
        {
            "name": "data_format",
            "param_type": "optional",
            "type": "str",
            "value": "all"
        },
        {
            "name": "is_training",
            "param_type": "optional",
            "type": "bool",
            "value": "all"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float16","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "y_backprop",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float16","float16","float16","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 2,
            "dtype": [
                "float","float","float","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "scale",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 3,
            "dtype": [
                "float","float","float","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "reserve_space_1",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 4,
            "dtype": [
                "float","float","float","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "reserve_space_2",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 5,
            "dtype": [
                "float","float","float","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "reserve_space_3",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float16","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "x_backprop",
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float","float","float","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "scale_backprop",
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 2,
            "dtype": [
                "float","float","float","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "offset_backprop",
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 3,
            "dtype": [
                "float","float","float","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "reserve_space_4",
            "param_type": "optional",
            "shape": "all"
        },
        {
            "index": 4,
            "dtype": [
                "float","float","float","float","float","float"
            ],
            "format": [
                "DefaultFormat","DefaultFormat","NC1HWC0","DefaultFormat","DefaultFormat","NC1HWC0"
            ],
            "name": "reserve_space_5",
            "param_type": "optional",
            "shape": "all"
        }
    ]
}""")
def _batch_norm_grad_tbe():
    """BatchNormGrad TBE register"""
    return
