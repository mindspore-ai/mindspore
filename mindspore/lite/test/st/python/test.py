# Copyright 2022 Huawei Technologies Co., Ltd
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
Test lite python API.
"""
import mindspore_lite as mslite
import numpy as np

gpu_device_info = mslite.context.GPUDeviceInfo(enable_fp16=False, device_id=8)
cpu_device_info = mslite.context.CPUDeviceInfo(enable_fp16=False)
print("gpu_device_info: ", gpu_device_info)
print("cpu_device_info: ", cpu_device_info)

context = mslite.context.Context(thread_num=1, thread_affinity_mode=2)

context.append_device_info(cpu_device_info)
print("context: ", context)

model = mslite.model.Model()
model.build_from_file("mnist.tflite.ms", mslite.model.ModelType.MINDIR_LITE, context)
print("model: ", model)

inputs = model.get_inputs()
outputs = model.get_outputs()
print("input num: ", len(inputs))
print("output num: ", len(outputs))

in_data = np.fromfile("mnist.tflite.ms.bin", dtype=float)
inputs[0].set_data_from_numpy(in_data)
print("input: ", inputs[0])

model.predict(inputs, outputs)

for output in outputs:
    print("output: ", output)
    data = output.get_data_to_numpy()
    print("data: ", data)
