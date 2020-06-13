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

"""SparseSoftmaxCrossEntropyWithLogits op"""
from mindspore.ops.op_info_register import op_info_register


@op_info_register("""{
    "op_name": "SparseSoftmaxCrossEntropyWithLogits",
    "imply_type": "AutoDiff",
    "fusion_type": "OPAQUE",
    "attr": [
        {
            "name": "is_grad",
            "param_type": "optional",
            "type": "bool"
        },
        {
            "name": "sens",
            "param_type": "optional",
            "type": "float"
        }
    ],
    "inputs": [
        {
            "index": 0,
            "dtype": [
                "float32"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "features"
        },
        {
            "index": 1,
            "dtype": [
                "int32"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "labels"
        }
    ],
    "outputs": [
        {
            "index": 0,
            "dtype": [
                "float32"
            ],
            "format": [
                "DefaultFormat"
            ],
            "name": "output"
        }
    ]
}""")
def _sparse_softmax_cross_entropy_with_logits_akg():
    """SparseSoftmaxCrossEntropyWithLogits AutoDiff register"""
    return
