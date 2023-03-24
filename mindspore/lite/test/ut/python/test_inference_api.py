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
Test lite inference python API.
"""
import mindspore_lite as mslite
import numpy as np
import pytest


# ============================ Context ============================
def test_context_construct():
    context = mslite.Context()
    assert "target:" in str(context)


def test_context_target_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.target = 1
    assert "target must be list" in str(raise_info.value)


def test_context_target_list_element_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.target = [1]
    assert "target element must be str" in str(raise_info.value)


def test_context_target_list_element_value_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context()
        context.target = ["1"]
    assert "target elements must be in" in str(raise_info.value)


def test_context_target():
    context = mslite.Context()
    context.target = ["cpu"]
    assert context.target == ["cpu"]
    context.target = ["gpu"]
    assert context.target == ["gpu"]
    context.target = ["ascend"]
    assert context.target == ["ascend"]
    context.target = []
    assert context.target == ["cpu"]


def test_context_cpu_precision_mode_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.cpu.precision_mode = 1
    assert "cpu_precision_mode must be str" in str(raise_info.value)


def test_context_cpu_precision_mode_value_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context()
        context.cpu.precision_mode = "1"
    assert "cpu_precision_mode must be in" in str(raise_info.value)


def test_context_cpu_precision_mode():
    context = mslite.Context()
    context.cpu.precision_mode = "preferred_fp16"
    assert "precision_mode: preferred_fp16" in str(context.cpu)


def test_context_cpu_thread_num_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.cpu.thread_num = "1"
    assert "cpu_thread_num must be int" in str(raise_info.value)


def test_context_cpu_thread_num_negative_value_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context()
        context.cpu.thread_num = -1
    assert "cpu_thread_num must be a non-negative int" in str(raise_info.value)


def test_context_cpu_thread_num():
    context = mslite.Context()
    context.cpu.thread_num = 4
    assert "thread_num: 4" in str(context.cpu)


def test_context_cpu_inter_op_parallel_num_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.cpu.inter_op_parallel_num = "1"
    assert "cpu_inter_op_parallel_num must be int" in str(raise_info.value)


def test_context_cpu_inter_op_parallel_num_negative_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context()
        context.cpu.inter_op_parallel_num = -1
    assert "cpu_inter_op_parallel_num must be a non-negative int" in str(raise_info.value)


def test_context_cpu_inter_op_parallel_num():
    context = mslite.Context()
    context.cpu.inter_op_parallel_num = 1
    assert "inter_op_parallel_num: 1" in str(context.cpu)


def test_context_cpu_thread_affinity_mode_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.cpu.thread_affinity_mode = "1"
    assert "cpu_thread_affinity_mode must be int" in str(raise_info.value)


def test_context_cpu_thread_affinity_mode():
    context = mslite.Context()
    context.cpu.thread_affinity_mode = 2
    assert "thread_affinity_mode: 2" in str(context.cpu)


def test_context_cpu_thread_affinity_core_list_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.cpu.thread_affinity_core_list = 2
    assert "cpu_thread_affinity_core_list must be list" in str(raise_info.value)


def test_context_cpu_thread_affinity_core_list_element_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.cpu.thread_affinity_core_list = ["1", "0"]
    assert "cpu_thread_affinity_core_list element must be int" in str(raise_info.value)


def test_context_cpu_thread_affinity_core_list():
    context = mslite.Context()
    context.cpu.thread_affinity_core_list = [2]
    assert "thread_affinity_core_list: [2]" in str(context.cpu)
    context.cpu.thread_affinity_core_list = [1, 0]
    assert "thread_affinity_core_list: [1, 0]" in str(context.cpu)


def test_context_gpu_precision_mode_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.gpu.precision_mode = 1
    assert "gpu_precision_mode must be str" in str(raise_info.value)


def test_context_gpu_precision_mode_value_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context()
        context.gpu.precision_mode = "1"
    assert "gpu_precision_mode must be in" in str(raise_info.value)


def test_context_gpu_precision_mode():
    context = mslite.Context()
    context.gpu.precision_mode = "preferred_fp16"
    assert "precision_mode: preferred_fp16" in str(context.gpu)


def test_context_gpu_device_id_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.gpu.device_id = "1"
    assert "gpu_device_id must be int" in str(raise_info.value)


def test_context_gpu_device_id_negative_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context()
        context.gpu.device_id = -1
    assert "gpu_device_id must be a non-negative int" in str(raise_info.value)


def test_context_gpu_device_id():
    context = mslite.Context()
    context.gpu.device_id = 1
    assert "device_id: 1" in str(context.gpu)


def test_context_ascend_precision_mode_value_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context()
        context.ascend.precision_mode = "1"
    assert "ascend_precision_mode must be in" in str(raise_info.value)


def test_context_ascend_precision_mode():
    context = mslite.Context()
    context.ascend.precision_mode = "enforce_fp32"
    assert "precision_mode: enforce_fp32" in str(context.ascend)


def test_context_ascend_device_id_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.ascend.device_id = "1"
    assert "ascend_device_id must be int" in str(raise_info.value)


def test_context_ascend_device_id_negative_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context()
        context.ascend.device_id = -1
    assert "ascend_device_id must be a non-negative int" in str(raise_info.value)


def test_context_ascend_device_id():
    context = mslite.Context()
    context.ascend.device_id = 1
    assert "device_id: 1" in str(context.ascend)


# ============================ Model ============================
def test_model_01():
    model = mslite.Model()
    assert "model_path:" in str(model)


def test_model_build_from_file_model_path_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = mslite.Model()
        model.build_from_file(model_path=1, model_type=mslite.ModelType.MINDIR_LITE)
    assert "model_path must be str" in str(raise_info.value)


def test_model_build_from_file_model_path_not_exist_error():
    with pytest.raises(RuntimeError) as raise_info:
        model = mslite.Model()
        model.build_from_file(model_path="test.ms", model_type=mslite.ModelType.MINDIR_LITE)
    assert "model_path does not exist" in str(raise_info.value)


def test_model_build_from_file_model_type_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = mslite.Model()
        model.build_from_file(model_path="test.ms", model_type="MINDIR_LITE")
    assert "model_type must be ModelType" in str(raise_info.value)


def test_model_build_from_file_context_type_error():
    with pytest.raises(TypeError) as raise_info:
        cpu_device_info = mslite.Context().cpu
        model = mslite.Model()
        model.build_from_file(model_path="test.ms", model_type=mslite.ModelType.MINDIR_LITE, context=cpu_device_info)
    assert "context must be Context" in str(raise_info.value)


def test_model_build_from_file_config_path_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = mslite.Model()
        model.build_from_file(model_path="mobilenetv2.ms", model_type=mslite.ModelType.MINDIR_LITE,
                              config_path=1)
    assert "config_path must be str" in str(raise_info.value)


def test_model_build_from_file_config_path_not_exist_error():
    with pytest.raises(RuntimeError) as raise_info:
        model = mslite.Model()
        model.build_from_file(model_path="mobilenetv2.ms", model_type=mslite.ModelType.MINDIR_LITE,
                              config_path="test.cfg")
    assert "config_path does not exist" in str(raise_info.value)


def get_model():
    context = mslite.Context()
    context.target = ["cpu"]
    context.cpu.thread_num = 2
    model = mslite.Model()
    model.build_from_file(model_path="mobilenetv2.ms", model_type=mslite.ModelType.MINDIR_LITE, context=context)
    return model


def test_model_resize_inputs_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        model.resize(inputs[0], [[1, 112, 112, 3]])
    assert "inputs must be list" in str(raise_info.value)


def test_model_resize_inputs_elements_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        model.resize([1, 2], [[1, 112, 112, 3]])
    assert "inputs element must be Tensor" in str(raise_info.value)


def test_model_resize_dims_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        model.resize(inputs, "[[1, 112, 112, 3]]")
    assert "dims must be list" in str(raise_info.value)


def test_model_resize_dims_elements_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        model.resize(inputs, ["[1, 112, 112, 3]"])
    assert "dims element must be list" in str(raise_info.value)


def test_model_resize_dims_elements_elements_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        model.resize(inputs, [[1, "112", 112, 3]])
    assert "dims element's element must be int" in str(raise_info.value)


def test_model_resize_inputs_size_not_equal_dims_size_error():
    with pytest.raises(ValueError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        model.resize(inputs, [[1, 112, 112, 3], [1, 112, 112, 3]])
    assert "inputs' size does not match dims' size" in str(raise_info.value)


def test_model_resize_01():
    model = get_model()
    inputs = model.get_inputs()
    assert inputs[0].shape == [1, 224, 224, 3]
    model.resize(inputs, [[1, 112, 112, 3]])
    assert inputs[0].shape == [1, 112, 112, 3]


def test_model_predict_inputs_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        outputs = model.predict(inputs[0])
    assert "inputs must be list" in str(raise_info.value)


def test_model_predict_inputs_element_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        outputs = model.predict(["input"])
    assert "inputs element must be Tensor" in str(raise_info.value)


def test_model_predict_runtime_error():
    with pytest.raises(RuntimeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        outputs = model.predict(inputs)
    assert "predict failed" in str(raise_info.value)


def test_model_predict_01():
    model = get_model()
    inputs = model.get_inputs()
    in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
    inputs[0].set_data_from_numpy(in_data)
    outputs = model.predict(inputs)


def test_model_predict_02():
    model = get_model()
    inputs = model.get_inputs()
    input_tensor = mslite.Tensor()
    input_tensor.dtype = inputs[0].dtype
    input_tensor.shape = inputs[0].shape
    input_tensor.format = inputs[0].format
    input_tensor.name = inputs[0].name
    in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
    input_tensor.set_data_from_numpy(in_data)
    outputs = model.predict([input_tensor])


# ============================ Tensor ============================
def test_tensor_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor1 = mslite.Tensor()
        tensor2 = mslite.Tensor(tensor=tensor1)
    assert "tensor must be MindSpore Lite's Tensor._tensor" in str(raise_info.value)


def test_tensor():
    tensor1 = mslite.Tensor()
    assert tensor1.name == ""


def test_tensor_name_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.name = 1
    assert "name must be str" in str(raise_info.value)


def test_tensor_name():
    tensor = mslite.Tensor()
    tensor.name = "tensor0"
    assert tensor.name == "tensor0"


def test_tensor_dtype_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.dtype = 1
    assert "dtype must be DataType" in str(raise_info.value)


def test_tensor_dtype():
    tensor = mslite.Tensor()
    tensor.dtype = mslite.DataType.INT32
    assert tensor.dtype == mslite.DataType.INT32


def test_tensor_shape_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.shape = 224
    assert "shape must be list" in str(raise_info.value)


def test_tensor_shape_element_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.shape = ["224", "224"]
    assert "shape element must be int" in str(raise_info.value)


def test_tensor_shape_get_element_num_get_data_size_01():
    tensor = mslite.Tensor()
    tensor.dtype = mslite.DataType.FLOAT32
    tensor.shape = [16, 16]
    assert tensor.shape == [16, 16]
    assert tensor.element_num == 256
    assert tensor.data_size == 1024


def test_tensor_format_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.format = 1
    assert "format must be Format" in str(raise_info.value)


def test_tensor_format():
    tensor = mslite.Tensor()
    tensor.format = mslite.Format.NHWC4
    assert tensor.format == mslite.Format.NHWC4


def test_tensor_set_data_from_numpy_numpy_obj_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.set_data_from_numpy(1)
    assert "numpy_obj must be numpy.ndarray," in str(raise_info.value)


def test_tensor_set_data_from_numpy_data_type_not_equal_error():
    with pytest.raises(RuntimeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.dtype = mslite.DataType.FLOAT32
        tensor.shape = [2, 3]
        in_data = np.arange(2 * 3, dtype=np.int32).reshape((2, 3))
        tensor.set_data_from_numpy(in_data)
    assert "data type not equal" in str(raise_info.value)


def test_tensor_set_data_from_numpy_data_size_not_equal_error():
    with pytest.raises(RuntimeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.dtype = mslite.DataType.FLOAT32
        in_data = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
        tensor.set_data_from_numpy(in_data)
    assert "data size not equal" in str(raise_info.value)


def test_tensor_set_data_from_numpy():
    tensor = mslite.Tensor()
    tensor.dtype = mslite.DataType.FLOAT32
    tensor.shape = [2, 3]
    in_data = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
    tensor.set_data_from_numpy(in_data)
    out_data = tensor.get_data_to_numpy()
    assert (out_data == in_data).all()
