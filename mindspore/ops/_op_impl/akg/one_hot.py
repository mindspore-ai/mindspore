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

"""OneHot op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "OneHot",
    "imply_type": "AutoDiff",
    "fusion_type": "OPAQUE",
    "attr": [
        {
            "name": "depth",
            "param_type": "required",
            "type": "int"
        },
        {
            "name": "axis",
            "param_type": "required",
            "type": "int"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "int32", "int32", "int32"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat", "DefaultFormat"
            ],
            "name": "indices"
        },
        {
            "index": 1,
            "dtype": [
                "int32", "float32", "float16"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat", "DefaultFormat"
            ],
            "name": "on_value"
        },
        {
            "index": 2,
            "dtype": [
                "int32", "float32", "float16"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat", "DefaultFormat"
            ],
            "name": "off_value"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "int32", "float32", "float16"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat", "DefaultFormat"
            ],
            "name": "output"
        }
    ]
}""")
def _one_hot_akg():
    """OneHot AutoDiff register"""
    return
