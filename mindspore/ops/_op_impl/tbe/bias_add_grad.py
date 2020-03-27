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

"""BiasAddGrad op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "BiasAddGrad",
    "imply_type": "TBE",
    "fusion_type": "COMMREDUCE",
    "async_flag": false,
    "binfile_name": "biasaddgrad.so",
    "compute_cost": 10,
    "kernel_name": "biasaddgrad",
    "partial_flag": true,
    "attr": [
        {
            "name": "data_format",
            "param_type": "required",
            "type": "str",
            "value": "all"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16","float16","float","float"
            ],
            "format": [
                "FRACTAL_NZ","DefaultFormat","FRACTAL_NZ","DefaultFormat"
            ],
            "name": "out_backprop",
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
                "DefaultFormat","DefaultFormat","DefaultFormat","DefaultFormat"
            ],
            "name": "output",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _bias_add_grad_tbe():
    """BiasAddGrad TBE register"""
    return
