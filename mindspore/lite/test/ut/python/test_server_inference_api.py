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


# ============================ Context.parallel ============================

def test_context_parallel_workers_num_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.parallel.workers_num = "4"
    assert "workers_num must be int" in str(raise_info.value)


def test_context_parallel_workers_num_negative_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context()
        context.parallel.workers_num = -4
    assert "workers_num must be a non-negative int" in str(raise_info.value)


def test_context_parallel_config_info_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.parallel.config_info = 1
    assert "config_info must be dict" in str(raise_info.value)


def test_context_parallel_config_info_key_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.parallel.config_info = {1: {"test": "test"}}
    assert "config_info_key must be str" in str(raise_info.value)


def test_context_parallel_config_info_value_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.parallel.config_info = {"test": "test"}
    assert "config_info_value must be dict" in str(raise_info.value)


def test_context_parallel_config_info_value_key_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.parallel.config_info = {"test": {1: "test"}}
    assert "config_info_value_key must be str" in str(raise_info.value)


def test_context_parallel_config_info_value_value_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.parallel.config_info = {"test": {"test": 1}}
    assert "config_info_value_value must be str" in str(raise_info.value)


def test_context_parallel_config_path_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.parallel.config_path = 1
    assert "config_path must be str" in str(raise_info.value)


def test_context_parallel_config_path_not_exist_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context()
        context.parallel.config_path = "test.cfg"
    assert "config_path does not exist" in str(raise_info.value)


def test_context_parallel():
    config_info = {"weight": {"weight_path": "path of model weight"}}
    context = mslite.Context()
    context.target = ["cpu"]
    context.parallel.workers_num = 4
    assert "workers num:" in str(context.parallel)
    context.parallel.config_info = config_info
    assert "config info:" in str(context.parallel)


# ============================ ModelParallelRunner ============================
def test_model_parallel_runner():
    model_parallel_runner = mslite.ModelParallelRunner()
    assert "model_path:" in str(model_parallel_runner)


def test_model_parallel_runner_build_from_file_model_path_type_error():
    with pytest.raises(TypeError) as raise_info:
        model_parallel_runner = mslite.ModelParallelRunner()
        model_parallel_runner.build_from_file(model_path=["test.ms"])
    assert "model_path must be str" in str(raise_info.value)


def test_model_parallel_runner_build_from_file_model_path_not_exist_error():
    with pytest.raises(RuntimeError) as raise_info:
        model_parallel_runner = mslite.ModelParallelRunner()
        model_parallel_runner.build_from_file(model_path="test.ms")
    assert "model_path does not exist" in str(raise_info.value)


def test_model_parallel_runner_build_from_file_01():
    model_parallel_runner = mslite.model.ModelParallelRunner()
    model_parallel_runner.build_from_file(model_path="mobilenetv2.ms")
    assert "model_path:" in str(model_parallel_runner)


def test_model_parallel_runner_build_from_file_02():
    context = mslite.Context()
    context.target = ["cpu"]
    context.parallel.workers_num = 4
    model_parallel_runner = mslite.model.ModelParallelRunner()
    model_parallel_runner.build_from_file(model_path="mobilenetv2.ms", context=context)
    assert "model_path:" in str(model_parallel_runner)


def get_model_parallel_runner():
    context = mslite.Context()
    context.target = ["cpu"]
    context.parallel.workers_num = 4
    model_parallel_runner = mslite.ModelParallelRunner()
    model_parallel_runner.build_from_file(model_path="mobilenetv2.ms", context=context)
    return model_parallel_runner


def test_model_parallel_runner_predict_inputs_type_error():
    with pytest.raises(TypeError) as raise_info:
        model_parallel_runner = get_model_parallel_runner()
        inputs = model_parallel_runner.get_inputs()
        outputs = model_parallel_runner.predict(inputs[0])
    assert "inputs must be list" in str(raise_info.value)


def test_model_parallel_runner_predict_inputs_elements_type_error():
    with pytest.raises(TypeError) as raise_info:
        model_parallel_runner = get_model_parallel_runner()
        inputs = model_parallel_runner.get_inputs()
        outputs = model_parallel_runner.predict(["input"])
    assert "inputs element must be Tensor" in str(raise_info.value)


def test_model_parallel_runner_predict_runtime_error():
    with pytest.raises(RuntimeError) as raise_info:
        model_parallel_runner = get_model_parallel_runner()
        tensor1 = mslite.Tensor()
        tensor2 = mslite.Tensor()
        inputs = [tensor1, tensor2]
        outputs = model_parallel_runner.predict(inputs)
    assert "predict failed" in str(raise_info.value)


def test_model_parallel_runner_predict_01():
    model_parallel_runner = get_model_parallel_runner()
    inputs = model_parallel_runner.get_inputs()
    in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
    inputs[0].set_data_from_numpy(in_data)
    outputs = model_parallel_runner.predict(inputs)


def test_model_parallel_runner_predict_02():
    model_parallel_runner = get_model_parallel_runner()
    inputs = model_parallel_runner.get_inputs()
    input_tensor = mslite.Tensor()
    input_tensor.dtype = inputs[0].dtype
    input_tensor.shape = inputs[0].shape
    input_tensor.format = inputs[0].format
    input_tensor.name = inputs[0].name
    in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
    input_tensor.set_data_from_numpy(in_data)
    outputs = model_parallel_runner.predict([input_tensor])
