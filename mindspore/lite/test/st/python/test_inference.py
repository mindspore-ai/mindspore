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
    in_data = np.fromfile(in_data_path, dtype=np.float32)
    inputs[0].set_data_from_numpy(in_data)
    outputs = model.predict(inputs)
    for output in outputs:
        data = output.get_data_to_numpy()
        print("data: ", data)


# ============================ cpu inference ============================
def test_cpu_inference_01():
    context = mslite.Context()
    context.target = ["cpu"]
    context.cpu.thread_num = 1
    context.cpu.thread_affinity_mode = 2
    cpu_model_path = "mobilenetv2.ms"
    cpu_in_data_path = "mobilenetv2.ms.bin"
    common_predict(context, cpu_model_path, cpu_in_data_path)


# ============================ gpu inference ============================
def test_gpu_inference_01():
    context = mslite.Context()
    context.target = ["gpu"]
    context.gpu.device_id = 0
    print("gpu: ", context.gpu)
    context.cpu.thread_num = 1
    context.cpu.thread_affinity_mode = 2
    print("cpu_backup: ", context.cpu)
    gpu_model_path = "mobilenetv2.ms"
    gpu_in_data_path = "mobilenetv2.ms.bin"
    common_predict(context, gpu_model_path, gpu_in_data_path)


# ============================ ascend inference ============================
def test_ascend_inference_01():
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = 0
    print("ascend: ", context.ascend)
    context.cpu.thread_num = 1
    context.cpu.thread_affinity_mode = 2
    print("cpu_backup: ", context.cpu)
    ascend_model_path = "mnist.tflite.ms"
    ascend_in_data_path = "mnist.tflite.ms.bin"
    common_predict(context, ascend_model_path, ascend_in_data_path)


# ============================ server inference ============================
def test_server_inference_01():
    context = mslite.Context()
    context.target = ["cpu"]
    context.cpu.thread_num = 4
    context.parallel.workers_num = 1
    model_parallel_runner = mslite.ModelParallelRunner()
    cpu_model_path = "mobilenetv2.ms"
    cpu_in_data_path = "mobilenetv2.ms.bin"
    model_parallel_runner.build_from_file(model_path=cpu_model_path, context=context)

    inputs = model_parallel_runner.get_inputs()
    in_data = np.fromfile(cpu_in_data_path, dtype=np.float32)
    inputs[0].set_data_from_numpy(in_data)
    outputs = model_parallel_runner.predict(inputs)
    data = outputs[0].get_data_to_numpy()
    print("data: ", data)
