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


def common_predict(context, model_path, in_data_path):
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR_LITE, context)

    inputs = model.get_inputs()
    outputs = model.get_outputs()
    in_data = np.fromfile(in_data_path, dtype=np.float32)
    inputs[0].set_data_from_numpy(in_data)
    model.predict(inputs, outputs)
    for output in outputs:
        data = output.get_data_to_numpy()
        print("data: ", data)


# ============================ cpu inference ============================
def test_cpu_inference_01():
    cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
    print("cpu_device_info: ", cpu_device_info)
    context = mslite.Context(thread_num=1, thread_affinity_mode=2)
    context.append_device_info(cpu_device_info)
    cpu_model_path = "mobilenetv2.ms"
    cpu_in_data_path = "mobilenetv2.ms.bin"
    common_predict(context, cpu_model_path, cpu_in_data_path)


# ============================ gpu inference ============================
def test_gpu_inference_01():
    gpu_device_info = mslite.GPUDeviceInfo(device_id=0, enable_fp16=False)
    print("gpu_device_info: ", gpu_device_info)
    cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
    print("cpu_device_info: ", cpu_device_info)
    context = mslite.Context(thread_num=1, thread_affinity_mode=2)
    context.append_device_info(gpu_device_info)
    context.append_device_info(cpu_device_info)
    gpu_model_path = "mobilenetv2.ms"
    gpu_in_data_path = "mobilenetv2.ms.bin"
    common_predict(context, gpu_model_path, gpu_in_data_path)


# ============================ ascend inference ============================
def test_ascend_inference_01():
    ascend_device_info = mslite.AscendDeviceInfo(device_id=0)
    print("ascend_device_info: ", ascend_device_info)
    cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
    print("cpu_device_info: ", cpu_device_info)
    context = mslite.Context(thread_num=1, thread_affinity_mode=2)
    context.append_device_info(ascend_device_info)
    context.append_device_info(cpu_device_info)
    ascend_model_path = "mnist.tflite.ms"
    ascend_in_data_path = "mnist.tflite.ms.bin"
    common_predict(context, ascend_model_path, ascend_in_data_path)


# ============================ server inference ============================
def test_server_inference_01():
    cpu_device_info = mslite.CPUDeviceInfo()
    print("cpu_device_info: ", cpu_device_info)
    context = mslite.Context(thread_num=4)
    context.append_device_info(cpu_device_info)
    runner_config = mslite.RunnerConfig(context, 4)
    model_parallel_runner = mslite.ModelParallelRunner()
    cpu_model_path = "mobilenetv2.ms"
    cpu_in_data_path = "mobilenetv2.ms.bin"
    model_parallel_runner.init(model_path=cpu_model_path, runner_config=runner_config)

    inputs = model_parallel_runner.get_inputs()
    in_data = np.fromfile(cpu_in_data_path, dtype=np.float32)
    inputs[0].set_data_from_numpy(in_data)
    outputs = model_parallel_runner.get_outputs()
    model_parallel_runner.predict(inputs, outputs)
    data = outputs[0].get_data_to_numpy()
    print("data: ", data)


# ============================ server inference ============================
def test_server_inference_02():
    cpu_device_info = mslite.CPUDeviceInfo()
    print("cpu_device_info: ", cpu_device_info)
    context = mslite.Context(thread_num=4)
    context.append_device_info(cpu_device_info)
    runner_config = mslite.RunnerConfig(context, 1)
    model_parallel_runner = mslite.ModelParallelRunner()
    cpu_model_path = "mobilenetv2.ms"
    cpu_in_data_path = "mobilenetv2.ms.bin"
    model_parallel_runner.init(model_path=cpu_model_path, runner_config=runner_config)

    inputs = model_parallel_runner.get_inputs()
    in_data = np.fromfile(cpu_in_data_path, dtype=np.float32)
    inputs[0].set_data_from_numpy(in_data)
    outputs = []
    model_parallel_runner.predict(inputs, outputs)
    data = outputs[0].get_data_to_numpy()
    print("data: ", data)
