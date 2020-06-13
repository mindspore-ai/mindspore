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

"""EquivFormat op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "EquivFormat",
    "imply_type": "AutoDiff",
    "fusion_type": "ELEMWISE",
    "attr": [

    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float32", "float16", "float32"
            ],
            "format": [
                "DefaultFormat", "DefaultFormat", "FRACTAL_NZ", "FRACTAL_NZ"
            ],
            "name": "x"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float32", "float16", "float32"
            ],
            "format": [
                "FRACTAL_NZ", "FRACTAL_NZ", "DefaultFormat", "DefaultFormat"
            ],
            "name": "output"
        }
    ]
}""")
def _equiv_format_akg():
    """EquivFormat AutoDiff register"""
    return
