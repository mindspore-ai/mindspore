# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""quick_start_cloud_infer_python."""

import numpy as np
import mindspore_lite as mslite


# init context, and set cpu target
context = mslite.Context()
context.target = ["cpu"]
context.cpu.thread_num = 1
context.cpu.thread_affinity_mode = 2
# build model from file
MODEL_PATH = "./model/mobilenetv2.mindir"
IN_DATA_PATH = "./model/input.bin"
model = mslite.Model()
model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR, context)
# set model input
inputs = model.get_inputs()
in_data = np.fromfile(IN_DATA_PATH, dtype=np.float32)
inputs[0].set_data_from_numpy(in_data)
# execute inference
outputs = model.predict(inputs)
# get output
for output in outputs:
    name = output.name.rstrip()
    data_size = output.data_size
    element_num = output.element_num
    print("tensor's name is:%s data size is:%s tensor elements num is:%s" % (name, data_size, element_num))
    data = output.get_data_to_numpy()
    data = data.flatten()
    print("output data is:", end=" ")
    for i in range(50):
        print(data[i], end=" ")
    print("")
