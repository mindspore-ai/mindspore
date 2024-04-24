# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
MindSpore Lite predict engine
"""

import mindspore_lite as mslite


class MSLitePredict:
    """
    MindSpore Lite predict core process.
    """

    def __init__(self, mindir_file, device_id):
        self.model = mslite.Model()
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = device_id
        self.model.build_from_file(mindir_file, mslite.ModelType.MINDIR, context)

    def run(self, input_data):
        result = []
        inputs = self.model.get_inputs()
        for i, _input in enumerate(inputs):
            _input.set_data_from_numpy(input_data[i])
        outputs = self.model.predict(inputs)
        for output in outputs:
            result.append(output.get_data_to_numpy())
        return result
