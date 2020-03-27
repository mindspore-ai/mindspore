# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed unde:q!r the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""SigmoidCrossEntropyWithLogits op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "SigmoidCrossEntropyWithLogits",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "sigmoid_cross_entropy_with_logits.so",
    "compute_cost": 10,
    "kernel_name": "sigmoid_cross_entropy_with_logits",
    "partial_flag": true,
    "attr": [
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float16", "float", "float"
            ],
            "format": [
                "NC1HWC0", "DefaultFormat", "NC1HWC0", "DefaultFormat"
            ],
            "name": "predict",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        },
        {
            "index": 1,
            "dtype": [
                "float16", "float16", "float", "float"
            ],
            "format": [
                "NC1HWC0", "DefaultFormat", "NC1HWC0", "DefaultFormat"
            ],
            "name": "target",
            "need_compile": false,
            "param_type": "required",
            "shape": "all"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float16", "float16", "float", "float"
            ],
            "format": [
                "NC1HWC0", "DefaultFormat", "NC1HWC0", "DefaultFormat"
            ],
            "name": "loss",
            "need_compile": true,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _sigmoid_cross_entropy_with_logits_tbe():
    """SigmoidCrossEntropyWithLogits TBE register"""
    return
