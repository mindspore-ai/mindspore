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


# ============================ CPUDeviceInfo ============================
def test_cpu_device_info_enable_fp16_type_error():
    with pytest.raises(TypeError) as raise_info:
        device_info = mslite.CPUDeviceInfo(enable_fp16="1")
    assert "enable_fp16 must be bool" in str(raise_info.value)


def test_cpu_device_info_01():
    device_info = mslite.CPUDeviceInfo(enable_fp16=True)
    assert "enable_fp16: True" in str(device_info)


# ============================ GPUDeviceInfo ============================
def test_gpu_device_info_device_id_type_error():
    with pytest.raises(TypeError) as raise_info:
        device_info = mslite.GPUDeviceInfo(device_id="1")
    assert "device_id must be int" in str(raise_info.value)


def test_gpu_device_info_device_id_negative_error():
    with pytest.raises(ValueError) as raise_info:
        device_info = mslite.GPUDeviceInfo(device_id=-1)
    assert "device_id must be a non-negative int" in str(raise_info.value)


def test_gpu_device_info_enable_fp16_type_error():
    with pytest.raises(TypeError) as raise_info:
        device_info = mslite.GPUDeviceInfo(enable_fp16=1)
    assert "enable_fp16 must be bool" in str(raise_info.value)


def test_gpu_device_info_01():
    device_info = mslite.GPUDeviceInfo(device_id=2)
    assert "device_id: 2" in str(device_info)


def test_gpu_device_info_02():
    device_info = mslite.GPUDeviceInfo(enable_fp16=True)
    assert "enable_fp16: True" in str(device_info)


def test_gpu_device_info_get_rank_id_01():
    device_info = mslite.GPUDeviceInfo()
    rank_id = device_info.get_rank_id()
    assert isinstance(rank_id, int)


def test_gpu_device_info_get_group_size_01():
    device_info = mslite.GPUDeviceInfo()
    group_size = device_info.get_group_size()
    assert isinstance(group_size, int)


# ============================ AscendDeviceInfo ============================
def test_ascend_device_info_device_id_type_error():
    with pytest.raises(TypeError) as raise_info:
        device_info = mslite.AscendDeviceInfo(device_id="1")
    assert "device_id must be int" in str(raise_info.value)


def test_ascend_device_info_device_id_negative_error():
    with pytest.raises(ValueError) as raise_info:
        device_info = mslite.AscendDeviceInfo(device_id=-1)
    assert "device_id must be a non-negative int" in str(raise_info.value)


def test_ascend_device_info_01():
    device_info = mslite.AscendDeviceInfo(device_id=1)
    assert "device_id: 1" in str(device_info)


# ============================ Context ============================
def test_context_thread_num_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context(thread_num="1")
    assert "thread_num must be int" in str(raise_info.value)


def test_context_thread_num_negative_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context(thread_num=-1)
    assert "thread_num must be a non-negative int" in str(raise_info.value)


def test_context_inter_op_parallel_num_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context(thread_num=2, inter_op_parallel_num="1")
    assert "inter_op_parallel_num must be int" in str(raise_info.value)


def test_context_inter_op_parallel_num_negative_error():
    with pytest.raises(ValueError) as raise_info:
        context = mslite.Context(thread_num=2, inter_op_parallel_num=-1)
    assert "inter_op_parallel_num must be a non-negative int" in str(raise_info.value)


def test_context_thread_affinity_mode_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context(thread_num=2, thread_affinity_mode="1")
    assert "thread_affinity_mode must be int" in str(raise_info.value)


def test_context_thread_affinity_core_list_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context(thread_num=2, thread_affinity_core_list=2)
    assert "thread_affinity_core_list must be list" in str(raise_info.value)


def test_context_thread_affinity_core_list_element_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context(thread_num=2, thread_affinity_core_list=["1", "0"])
    assert "thread_affinity_core_list element must be int" in str(raise_info.value)


def test_context_enable_parallel_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context(thread_num=2, enable_parallel=1)
    assert "enable_parallel must be bool" in str(raise_info.value)


def test_context_01():
    context = mslite.Context()
    assert "thread_num:" in str(context)


def test_context_02():
    context = mslite.Context(thread_num=4)
    assert "thread_num: 4" in str(context)


def test_context_03():
    context = mslite.Context(thread_num=2, inter_op_parallel_num=1)
    assert "inter_op_parallel_num: 1" in str(context)


def test_context_04():
    context = mslite.Context(thread_num=2, thread_affinity_mode=2)
    assert "thread_affinity_mode: 2" in str(context)


def test_context_05():
    context = mslite.Context(thread_num=2, thread_affinity_core_list=[2])
    assert "thread_affinity_core_list: [2]" in str(context)


def test_context_06():
    context = mslite.Context(thread_num=2, thread_affinity_core_list=[1, 0])
    assert "thread_affinity_core_list: [1, 0]" in str(context)


def test_context_07():
    context = mslite.Context(thread_num=2, enable_parallel=True)
    assert "enable_parallel: True" in str(context)


def test_context_append_device_info_device_info_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context(thread_num=2)
        context.append_device_info("CPUDeviceInfo")
    assert "device_info must be DeviceInfo" in str(raise_info.value)


def test_context_append_device_info_01():
    cpu_device_info = mslite.CPUDeviceInfo()
    context = mslite.Context(thread_num=2)
    context.append_device_info(cpu_device_info)
    assert "device_list: 0" in str(context)


def test_context_append_device_info_02():
    gpu_device_info = mslite.GPUDeviceInfo()
    cpu_device_info = mslite.CPUDeviceInfo()
    context = mslite.Context(thread_num=2)
    context.append_device_info(gpu_device_info)
    context.append_device_info(cpu_device_info)
    assert "device_list: 1, 0" in str(context)


def test_context_append_device_info_03():
    ascend_device_info = mslite.AscendDeviceInfo()
    cpu_device_info = mslite.CPUDeviceInfo()
    context = mslite.Context(thread_num=2)
    context.append_device_info(ascend_device_info)
    context.append_device_info(cpu_device_info)
    assert "device_list: 3, 0" in str(context)


# ============================ Tensor ============================
def test_tensor_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor1 = mslite.Tensor()
        tensor2 = mslite.Tensor(tensor=tensor1)
    assert "tensor must be MindSpore Lite's Tensor" in str(raise_info.value)


def test_tensor_01():
    tensor = mslite.Tensor()
    assert tensor.get_tensor_name() == ""


def test_tensor_set_tensor_name_tensor_name_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.set_tensor_name(1)
    assert "tensor_name must be str" in str(raise_info.value)


def test_tensor_set_tensor_name_01():
    tensor = mslite.Tensor()
    tensor.set_tensor_name("tensor0")
    assert tensor.get_tensor_name() == "tensor0"


def test_tensor_set_data_type_data_type_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.set_data_type(1)
    assert "data_type must be DataType" in str(raise_info.value)


def test_tensor_set_data_type_01():
    tensor = mslite.Tensor()
    tensor.set_data_type(mslite.DataType.INT32)
    assert tensor.get_data_type() == mslite.DataType.INT32


def test_tensor_set_shape_shape_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.set_shape(224)
    assert "shape must be list" in str(raise_info.value)


def test_tensor_set_shape_shape_element_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.set_shape(["224", "224"])
    assert "shape element must be int" in str(raise_info.value)


def test_tensor_get_shape_get_element_num_get_data_size_01():
    tensor = mslite.Tensor()
    tensor.set_data_type(mslite.DataType.FLOAT32)
    tensor.set_shape([16, 16])
    assert tensor.get_shape() == [16, 16]
    assert tensor.get_element_num() == 256
    assert tensor.get_data_size() == 1024


def test_tensor_set_format_tensor_format_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.set_format(1)
    assert "tensor_format must be Format" in str(raise_info.value)


def test_tensor_set_format_01():
    tensor = mslite.Tensor()
    tensor.set_format(mslite.Format.NHWC4)
    assert tensor.get_format() == mslite.Format.NHWC4


def test_tensor_set_data_from_numpy_numpy_obj_type_error():
    with pytest.raises(TypeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.set_data_from_numpy(1)
    assert "numpy_obj must be numpy.ndarray," in str(raise_info.value)


def test_tensor_set_data_from_numpy_data_type_not_equal_error():
    with pytest.raises(RuntimeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.set_data_type(mslite.DataType.FLOAT32)
        tensor.set_shape([2, 3])
        in_data = np.arange(2 * 3, dtype=np.int32).reshape((2, 3))
        tensor.set_data_from_numpy(in_data)
    assert "data type not equal" in str(raise_info.value)


def test_tensor_set_data_from_numpy_data_size_not_equal_error():
    with pytest.raises(RuntimeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.set_data_type(mslite.DataType.FLOAT32)
        in_data = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
        tensor.set_data_from_numpy(in_data)
    assert "data size not equal" in str(raise_info.value)


def test_tensor_set_data_from_numpy_01():
    tensor = mslite.Tensor()
    tensor.set_data_type(mslite.DataType.FLOAT32)
    tensor.set_shape([2, 3])
    in_data = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
    tensor.set_data_from_numpy(in_data)
    out_data = tensor.get_data_to_numpy()
    assert (out_data == in_data).all()


# ============================ Model ============================
def test_model_01():
    model = mslite.Model()
    assert "model_path:" in str(model)


def test_model_build_from_file_model_path_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context(thread_num=2)
        model = mslite.Model()
        model.build_from_file(model_path=1, model_type=mslite.ModelType.MINDIR_LITE, context=context)
    assert "model_path must be str" in str(raise_info.value)


def test_model_build_from_file_model_path_not_exist_error():
    with pytest.raises(RuntimeError) as raise_info:
        context = mslite.Context(thread_num=2)
        model = mslite.Model()
        model.build_from_file(model_path="test.ms", model_type=mslite.ModelType.MINDIR_LITE, context=context)
    assert "model_path does not exist" in str(raise_info.value)


def test_model_build_from_file_model_type_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        model = mslite.Model()
        model.build_from_file(model_path="test.ms", model_type="MINDIR_LITE", context=context)
    assert "model_type must be ModelType" in str(raise_info.value)


def test_model_build_from_file_context_type_error():
    with pytest.raises(TypeError) as raise_info:
        device_info = mslite.CPUDeviceInfo()
        model = mslite.Model()
        model.build_from_file(model_path="test.ms", model_type=mslite.ModelType.MINDIR_LITE, context=device_info)
    assert "context must be Context" in str(raise_info.value)


def test_model_build_from_file_config_path_type_error():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context(thread_num=2)
        model = mslite.Model()
        model.build_from_file(model_path="mobilenetv2.ms", model_type=mslite.ModelType.MINDIR_LITE, context=context,
                              config_path=1)
    assert "config_path must be str" in str(raise_info.value)


def test_model_build_from_file_config_path_not_exist_error():
    with pytest.raises(RuntimeError) as raise_info:
        context = mslite.Context(thread_num=2)
        model = mslite.Model()
        model.build_from_file(model_path="mobilenetv2.ms", model_type=mslite.ModelType.MINDIR_LITE, context=context,
                              config_path="test.cfg")
    assert "config_path does not exist" in str(raise_info.value)


def get_model():
    cpu_device_info = mslite.CPUDeviceInfo()
    context = mslite.Context(thread_num=2)
    context.append_device_info(cpu_device_info)
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
    assert inputs[0].get_shape() == [1, 224, 224, 3]
    model.resize(inputs, [[1, 112, 112, 3]])
    assert inputs[0].get_shape() == [1, 112, 112, 3]


def test_model_predict_inputs_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        outputs = model.get_outputs()
        model.predict(inputs[0], outputs)
    assert "inputs must be list" in str(raise_info.value)


def test_model_predict_inputs_element_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        outputs = model.get_outputs()
        model.predict(["input"], outputs)
    assert "inputs element must be Tensor" in str(raise_info.value)


def test_model_predict_outputs_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        outputs = model.get_outputs()
        model.predict(inputs, outputs[0])
    assert "outputs must be list" in str(raise_info.value)


def test_model_predict_outputs_element_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        outputs = model.get_outputs()
        model.predict(inputs, ["output"])
    assert "outputs element must be Tensor" in str(raise_info.value)


def test_model_predict_runtime_error():
    with pytest.raises(RuntimeError) as raise_info:
        model = get_model()
        inputs = model.get_inputs()
        outputs = model.get_outputs()
        model.predict(inputs, outputs)
    assert "predict failed" in str(raise_info.value)


def test_model_predict_01():
    model = get_model()
    inputs = model.get_inputs()
    in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
    inputs[0].set_data_from_numpy(in_data)
    outputs = model.get_outputs()
    model.predict(inputs, outputs)


def test_model_predict_02():
    model = get_model()
    inputs = model.get_inputs()
    input_tensor = mslite.Tensor()
    input_tensor.set_data_type(inputs[0].get_data_type())
    input_tensor.set_shape(inputs[0].get_shape())
    input_tensor.set_format(inputs[0].get_format())
    input_tensor.set_tensor_name(inputs[0].get_tensor_name())
    in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
    input_tensor.set_data_from_numpy(in_data)
    outputs = model.get_outputs()
    model.predict([input_tensor], outputs)


def test_model_get_input_by_tensor_name_tensor_name_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        input_tensor = model.get_input_by_tensor_name(0)
    assert "tensor_name must be str" in str(raise_info.value)


def test_model_get_input_by_tensor_name_runtime_error():
    with pytest.raises(RuntimeError) as raise_info:
        model = get_model()
        input_tensor = model.get_input_by_tensor_name("no-exist")
    assert "get_input_by_tensor_name failed" in str(raise_info.value)


def test_model_get_input_by_tensor_name_01():
    model = get_model()
    input_tensor = model.get_input_by_tensor_name("graph_input-173")
    assert "tensor_name: graph_input-173" in str(input_tensor)


def test_model_get_output_by_tensor_name_tensor_name_type_error():
    with pytest.raises(TypeError) as raise_info:
        model = get_model()
        output = model.get_output_by_tensor_name(0)
    assert "tensor_name must be str" in str(raise_info.value)


def test_model_get_output_by_tensor_name_runtime_error():
    with pytest.raises(RuntimeError) as raise_info:
        model = get_model()
        output = model.get_output_by_tensor_name("no-exist")
    assert "get_output_by_tensor_name failed" in str(raise_info.value)


def test_model_get_output_by_tensor_name_01():
    model = get_model()
    output = model.get_output_by_tensor_name("Softmax-65")
    assert "tensor_name: Softmax-65" in str(output)
