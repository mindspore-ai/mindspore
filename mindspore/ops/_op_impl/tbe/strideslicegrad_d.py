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

"""StridedSliceGrad op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "StridedSliceGrad",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "strided_slice_grad_d.so",
    "compute_cost": 10,
    "kernel_name": "strided_slice_grad_d",
    "partial_flag": true,
    "attr": [
        {
            "name": "shapex",
            "param_type": "optional",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "begin",
            "param_type": "optional",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "end",
            "param_type": "optional",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "strides",
            "param_type": "optional",
            "type": "listInt",
            "value": "all"
        },
        {
            "name": "begin_mask",
            "param_type": "optional",
            "type": "int",
            "value": "all"
        },
        {
            "name": "end_mask",
            "param_type": "optional",
            "type": "int",
            "value": "all"
        },
        {
            "name": "ellipsis_mask",
            "param_type": "optional",
            "type": "int",
            "value": "all"
        },
        {
            "name": "new_axis_mask",
            "param_type": "optional",
            "type": "int",
            "value": "all"
        },
        {
            "name": "shrink_axis_mask",
            "param_type": "optional",
            "type": "int",
            "value": "all"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float","float","int32","int32","uint8","uint8","int8","int8"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0",
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0"
            ],
            "name": "dy",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float","float","int32","int32","uint8","uint8","int8","int8"
            ],
            "format": [
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0",
                "DefaultFormat","NC1HWC0","DefaultFormat","NC1HWC0"
            ],
            "name": "output",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _strided_slice_grad_d_tbe():
    """StridedSliceGrad TBE register"""
    return
