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

"""Five2Four op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "Five2Four",
    "imply_type": "AutoDiff",
    "fusion_type": "OPAQUE",
    "attr": [
        {
            "name": "shape4d",
            "param_type": "required",
            "type": "listInt"
        },
        {
            "name": "dstType",
            "param_type": "required",
            "type": "str"
        },
        {
            "name": "output_format",
            "param_type": "required",
            "type": "str"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float16","float32","float16","float32"
            ],
            "format": [
                "NC1HWC0","NC1HWC0","NC1HWC0","NC1HWC0","NC1HWC0","NC1HWC0"
            ],
            "name": "x"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float32","float32","float32","float32"
            ],
            "format": [
                "DefaultFormat","NHWC","DefaultFormat","DefaultFormat","NHWC","NHWC"
            ],
            "name": "output"
        }
    ]
}""")
def _five2four_akg():
    """Five2Four AutoDiff register"""
    return
