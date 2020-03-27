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

"""SoftmaxCrossEntropyWithLogits op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "SoftmaxCrossEntropyWithLogits",
    "imply_type": "TBE",
    "fusion_type": "OPAQUE",
    "async_flag": false,
    "binfile_name": "softmax_cross_entropy_with_logits.so",
    "compute_cost": 10,
    "kernel_name": "softmax_cross_entropy_with_logits",
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
            "name": "input_features",
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
            "name": "input_labels",
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
            "name": "output_loss",
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
            "name": "output_backprop",
            "need_compile": true,
            "param_type": "required",
            "shape": "all"
        }
    ]
}""")
def _softmax_cross_entropy_with_logits_tbe():
    """SoftmaxCrossEntropyWithLogits TBE register"""
    return
