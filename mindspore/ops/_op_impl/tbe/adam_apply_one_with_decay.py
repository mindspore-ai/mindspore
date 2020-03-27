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

"""AdamApplyOneWithDecay op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "AdamApplyOneWithDecay",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "adam_apply_one_with_decay.so",
    "compute_cost": 10,
    "kernel_name": "adam_apply_one_with_decay",
    "partial_flag": true,
    "attr": [

    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "input0",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "input1",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 2,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "input2",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 3,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "input3",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 4,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "input4",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 5,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "mul0_x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 6,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "mul1_x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 7,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "mul2_x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 8,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "mul3_x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 9,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "mul4_x",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 10,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "add2_y",
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
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "output0",
            "need_compile": true,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "output1",
            "need_compile": true,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 2,
            "dtype": [
                "float16", "float"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat"
            ],
            "name": "output2",
            "need_compile": true,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _adam_apply_one_with_decay_tbe():
    """AdamApplyOneWithDecay TBE register"""
    return
