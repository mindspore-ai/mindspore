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
Test lite server inference python API.
"""
import mindspore_lite as mslite
import numpy as np
import pytest


# ============================ RunnerConfig ============================
def test_runner_config_context_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        runner_config = mslite.RunnerConfig(context=cpu_device_info, workers_num=4, config_info=None)
    assert "context must be Context" in str(raise_info.value)


def test_runner_config_workers_num_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(context=context, workers_num="4", config_info=None)
    assert "workers_num must be int" in str(raise_info.value)


def test_runner_config_workers_num_negative_error():
    with pytest.raises(ValueError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(context=context, workers_num=-4, config_info=None)
    assert "workers_num must be a non-negative int" in str(raise_info.value)


def test_runner_config_config_info_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(context=context, workers_num=None, config_info=1)
    assert "config_info must be dict" in str(raise_info.value)


def test_runner_config_config_info_key_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(context=context, workers_num=None, config_info={1: {"test": "test"}})
    assert "config_info_key must be str" in str(raise_info.value)


def test_runner_config_config_info_value_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(context=context, workers_num=None, config_info={"test": "test"})
    assert "config_info_value must be dict" in str(raise_info.value)


def test_runner_config_config_info_value_key_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(context=context, workers_num=None, config_info={"test": {1: "test"}})
    assert "config_info_value_key must be str" in str(raise_info.value)


def test_runner_config_config_info_value_value_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(context=context, workers_num=None, config_info={"test": {"test": 1}})
    assert "config_info_value_value must be str" in str(raise_info.value)


def test_runner_config_config_path_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(config_path=1)
    assert "config_path must be str" in str(raise_info.value)


def test_runner_config_config_path_not_exist_error():
    with pytest.raises(RuntimeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(config_path="test.cfg")
    assert "config_path does not exist" in str(raise_info.value)


def test_runner_config_01():
    cpu_device_info = mslite.CPUDeviceInfo()
    context = mslite.Context()
    context.append_device_info(cpu_device_info)
    runner_config = mslite.RunnerConfig(context=context, workers_num=4, config_info=None)
    assert "workers num:" in str(runner_config)


def test_runner_config_02():
    cpu_device_info = mslite.CPUDeviceInfo()
    context = mslite.Context()
    context.append_device_info(cpu_device_info)
    config_info = {"weight": {"weight_path": "path of model weight"}}
    runner_config = mslite.RunnerConfig(context=context, workers_num=4, config_info=config_info)
    assert "config info:" in str(runner_config)


# ============================ ModelParallelRunner ============================
def test_model_parallel_runner_01():
    model_parallel_runner = mslite.ModelParallelRunner()
    assert "model_path:" in str(model_parallel_runner)


def test_model_parallel_runner_init_model_path_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(context, 4)
        model_parallel_runner = mslite.ModelParallelRunner()
        model_parallel_runner.init(model_path=["test.ms"], runner_config=runner_config)
    assert "model_path must be str" in str(raise_info.value)


def test_model_parallel_runner_init_model_path_not_exist_error():
    with pytest.raises(RuntimeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        runner_config = mslite.RunnerConfig(context, 4)
        model_parallel_runner = mslite.ModelParallelRunner()
        model_parallel_runner.init(model_path="test.ms", runner_config=runner_config)
    assert "model_path does not exist" in str(raise_info.value)


def test_model_parallel_runner_init_runner_config_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.CPUDeviceInfo()
        context = mslite.Context()
        context.append_device_info(cpu_device_info)
        model_parallel_runner = mslite.ModelParallelRunner()
        model_parallel_runner.init(model_path="mobilenetv2.ms", runner_config=context)
    assert "runner_config must be RunnerConfig" in str(raise_info.value)


def test_model_parallel_runner_init_runtime_error():
    with pytest.raises(RuntimeError) as raise_info:
        context = mslite.Context()
        runner_config = mslite.model.RunnerConfig(context, 4)
        model_parallel_runner = mslite.model.ModelParallelRunner()
        model_parallel_runner.init(model_path="mobilenetv2.ms", runner_config=runner_config)
    assert "init failed" in str(raise_info.value)


def test_model_parallel_runner_init_02():
    context = mslite.Context()
    model_parallel_runner = mslite.model.ModelParallelRunner()
    model_parallel_runner.init(model_path="mobilenetv2.ms")
    assert "model_path:" in str(model_parallel_runner)


def get_model_parallel_runner():
    cpu_device_info = mslite.CPUDeviceInfo()
    context = mslite.Context()
    context.append_device_info(cpu_device_info)
    runner_config = mslite.RunnerConfig(context, 4)
    model_parallel_runner = mslite.ModelParallelRunner()
    model_parallel_runner.init(model_path="mobilenetv2.ms", runner_config=runner_config)
    return model_parallel_runner


def test_model_parallel_runner_predict_inputs_type_error():
    with pytest.raises(TypeError) as raise_info:
        model_parallel_runner = get_model_parallel_runner()
        inputs = model_parallel_runner.get_inputs()
        outputs = model_parallel_runner.get_outputs()
        model_parallel_runner.predict(inputs[0], outputs)
    assert "inputs must be list" in str(raise_info.value)


def test_model_parallel_runner_predict_inputs_elements_type_error():
    with pytest.raises(TypeError) as raise_info:
        model_parallel_runner = get_model_parallel_runner()
        inputs = model_parallel_runner.get_inputs()
        outputs = model_parallel_runner.get_outputs()
        model_parallel_runner.predict(["input"], outputs)
    assert "inputs element must be Tensor" in str(raise_info.value)


def test_model_parallel_runner_predict_outputs_type_error():
    with pytest.raises(TypeError) as raise_info:
        model_parallel_runner = get_model_parallel_runner()
        inputs = model_parallel_runner.get_inputs()
        outputs = model_parallel_runner.get_outputs()
        model_parallel_runner.predict(inputs, outputs[0])
    assert "outputs must be list" in str(raise_info.value)


def test_model_parallel_runner_predict_outputs_elements_type_error():
    with pytest.raises(TypeError) as raise_info:
        model_parallel_runner = get_model_parallel_runner()
        inputs = model_parallel_runner.get_inputs()
        outputs = model_parallel_runner.get_outputs()
        model_parallel_runner.predict(inputs, ["output"])
    assert "outputs element must be Tensor" in str(raise_info.value)


def test_model_parallel_runner_predict_runtime_error():
    with pytest.raises(RuntimeError) as raise_info:
        model_parallel_runner = get_model_parallel_runner()
        tensor1 = mslite.Tensor()
        tensor2 = mslite.Tensor()
        inputs = [tensor1, tensor2]
        outputs = model_parallel_runner.get_outputs()
        model_parallel_runner.predict(inputs, outputs)
    assert "predict failed" in str(raise_info.value)


def test_model_parallel_runner_predict_01():
    model_parallel_runner = get_model_parallel_runner()
    inputs = model_parallel_runner.get_inputs()
    in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
    inputs[0].set_data_from_numpy(in_data)
    outputs = model_parallel_runner.get_outputs()
    model_parallel_runner.predict(inputs, outputs)


def test_model_parallel_runner_predict_02():
    model_parallel_runner = get_model_parallel_runner()
    inputs = model_parallel_runner.get_inputs()
    input_tensor = mslite.Tensor()
    input_tensor.set_data_type(inputs[0].get_data_type())
    input_tensor.set_shape(inputs[0].get_shape())
    input_tensor.set_format(inputs[0].get_format())
    in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
    input_tensor.set_data_from_numpy(in_data)
    outputs = model_parallel_runner.get_outputs()
    model_parallel_runner.predict([input_tensor], outputs)
