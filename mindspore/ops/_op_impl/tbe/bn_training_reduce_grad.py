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
    "op_name": "BNTrainingReduceGrad",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "bn_training_reduce_grad.so",
    "compute_cost": 10,
    "kernel_name": "bn_training_reduce_grad",
    "partial_flag": true,
    "attr": [
        {
            "name": "epsilon",
            "param_type": "optional",
            "type": "float",
            "value": "all"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "NC1HWC0","NC1HWC0"
            ],
            "name": "grads",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float", "float"
            ],
            "format": [
                "NC1HWC0","NC1HWC0"
            ],
            "name": "x_norm",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 2,
            "dtype": [
                "float", "float"
            ],
            "format": [
                "NC1HWC0","NC1HWC0"
            ],
            "name": "diff_scale",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 3,
            "dtype": [
                "float", "float"
            ],
            "format": [
                "NC1HWC0","NC1HWC0"
            ],
            "name": "diff_offset",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 4,
            "dtype": [
                "float", "float"
            ],
            "format": [
                "NC1HWC0","NC1HWC0"
            ],
            "name": "scale",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 5,
            "dtype": [
                "float", "float"
            ],
            "format": [
                "NC1HWC0","NC1HWC0"
            ],
            "name": "batch_mean",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 6,
            "dtype": [
                "float", "float"
            ],
            "format": [
                "NC1HWC0","NC1HWC0"
            ],
            "name": "batch_variance",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "NC1HWC0","NC1HWC0"
            ],
            "name": "y",
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _bn_training_reduce_grad_tbe():
    """BNTrainingReduceGrad TBE register"""
    return
