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
import sys
import os
import mindspore_lite as mslite
import numpy as np


def prepare_inputs_data(inputs, file_path, input_shapes):
    if file_path[0] == "random":
        print("use random data as input data.")
        for i in range(len(inputs)):
            dtype = inputs[i].dtype
            if dtype == mslite.DataType.FLOAT32:
                random_data = np.random.random(input_shapes[i]).astype(np.float32)
            elif dtype == mslite.DataType.INT32:
                random_data = np.random.random(input_shapes[i]).astype(np.int32)
            elif dtype == mslite.DataType.INT64:
                random_data = np.random.random(input_shapes[i]).astype(np.int64)
            else:
                raise RuntimeError('not support DataType!')
            inputs[i].shape = input_shapes[i]
            inputs[i].set_data_from_numpy(random_data)
        return
    for i in range(len(inputs)):
        dtype = inputs[i].dtype
        if dtype == mslite.DataType.FLOAT32:
            in_data = np.fromfile(file_path[i], dtype=np.float32)
        elif dtype == mslite.DataType.INT32:
            in_data = np.fromfile(file_path[i], dtype=np.int32)
        elif dtype == mslite.DataType.INT64:
            in_data = np.fromfile(file_path[i], dtype=np.int64)
        else:
            raise RuntimeError('not support DataType!')
        inputs[i].shape = input_shapes[i]
        inputs[i].set_data_from_numpy(in_data)


def get_inputs_group2_if_exist(in_data_path):
    in_data_path_group2 = []
    for data_path in in_data_path:
        data_path_list = data_path.rsplit('.', 1)
        data_path_list[-1] = "group2." + data_path_list[-1]
        data_path_new = '.'.join(data_path_list)
        if not os.path.exists(data_path_new):
            break
        in_data_path_group2.append(data_path_new)
    return in_data_path_group2


def model_common_predict(context, model_path, in_data_path, input_shapes):
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
    inputs = model.get_inputs()
    for input_tensor in inputs:
        input_shape = input_tensor.shape
        if -1 in input_shape:
            model.resize(inputs, input_shapes)
            break
    prepare_inputs_data(inputs, in_data_path, input_shapes)
    outputs = model.predict(inputs)
    in_data_path_group2 = get_inputs_group2_if_exist(in_data_path)
    if len(in_data_path_group2) == len(in_data_path):
        prepare_inputs_data(inputs, in_data_path_group2, input_shapes)
        outputs = model.predict(inputs)


def runner_common_predict(context, model_path, in_data_path, input_shapes):
    runner = mslite.ModelParallelRunner()
    runner.build_from_file(model_path=model_path, context=context)
    inputs = runner.get_inputs()
    prepare_inputs_data(inputs, in_data_path, input_shapes)
    outputs = runner.predict(inputs)
    in_data_path_group2 = get_inputs_group2_if_exist(in_data_path)
    if len(in_data_path_group2) == len(in_data_path):
        inputs = runner.get_inputs()
        prepare_inputs_data(inputs, in_data_path_group2, input_shapes)
        outputs = runner.predict(inputs)


# ============================ cpu inference ============================
def test_model_inference_cpu(model_path, in_data_path, input_shapes):
    context = mslite.Context()
    context.target = ["cpu"]
    context.cpu.thread_num = 2
    context.cpu.thread_affinity_mode = 0
    model_common_predict(context, model_path, in_data_path, input_shapes)


# ============================ gpu inference ============================
def test_model_inference_gpu(model_path, in_data_path, input_shapes):
    context = mslite.Context()
    context.target = ["gpu"]
    context.gpu.device_id = 0
    context.cpu.thread_num = 2
    context.cpu.thread_affinity_mode = 0
    model_common_predict(context, model_path, in_data_path, input_shapes)


# ============================ ascend inference ============================
def test_model_inference_ascend(model_path, in_data_path, input_shapes):
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = 0
    context.cpu.thread_num = 2
    context.cpu.thread_affinity_mode = 0
    model_common_predict(context, model_path, in_data_path, input_shapes)


# ============================ cpu server inference ============================
def test_parallel_inference_cpu(model_path, in_data_path, input_shapes):
    context = mslite.Context()
    context.target = ["cpu"]
    context.cpu.thread_num = 2
    context.cpu.thread_affinity_mode = 0
    context.parallel.workers_num = 2
    runner_common_predict(context, model_path, in_data_path, input_shapes)


# ============================ gpu server inference ============================
def test_parallel_inference_gpu(model_path, in_data_path, input_shapes):
    context = mslite.Context()
    context.target = ["gpu"]
    context.gpu.device_id = 0
    context.cpu.thread_num = 2
    context.cpu.thread_affinity_mode = 0
    context.parallel.workers_num = 2
    runner_common_predict(context, model_path, in_data_path, input_shapes)


# ============================ ascend server inference ============================
def test_parallel_inference_ascend(model_path, in_data_path, input_shapes):
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = 0
    context.cpu.thread_num = 2
    context.cpu.thread_affinity_mode = 0
    context.parallel.workers_num = 2
    runner_common_predict(context, model_path, in_data_path, input_shapes)


def test_model_group_for_ascend(model_path, in_data_path, input_shapes):
    # init model group context
    model_group_context = mslite.Context()
    model_group_context.target = ["ascend"]
    model_group_context.ascend.device_id = 0
    # init model group
    model_group = mslite.ModelGroup()
    model_group.add_model([model_path, model_path])  # test model group api for same model file.
    model_group.cal_max_size_of_workspace(mslite.ModelType.MINDIR, model_group_context)
    # use model one for inference
    test_model_inference_ascend(model_file, in_data_file_list, shapes)
    # use model two for inference
    test_model_inference_ascend(model_file, in_data_file_list, shapes)

def test_input_shape_for_ascend(model_path, input_shape_str):
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = 0
    context.cpu.thread_num = 2
    context.cpu.thread_affinity_mode = 0
    context.parallel.workers_num = 2
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
    input_shape_config = model.get_model_info("input_shape")

def test_model_group_weight_workspace_for_ascend(model_path, in_data_path, input_shapes):
    # init model group context
    model_group_context = mslite.Context()
    model_group_context.target = ["ascend"]
    model_group_context.ascend.device_id = 0
    # init model group
    model_group = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT_WORKSPACE)
    model_group.add_model([model_path, model_path])  # test model group api for same model file.
    model_group.cal_max_size_of_workspace(mslite.ModelType.MINDIR, model_group_context)
    # use model one for inference
    test_model_inference_ascend(model_file, in_data_file_list, shapes)


if __name__ == '__main__':
    model_file = sys.argv[1]
    in_data_file = sys.argv[2]
    input_shapes_str = sys.argv[3]
    backend = sys.argv[4]
    shapes = []
    input_shapes_list = input_shapes_str.split(":")
    for input_dims in input_shapes_list:
        dim = []
        dims = input_dims.split(",")
        for d in dims:
            dim.append(int(d))
        shapes.append(dim)
    in_data_file_list = in_data_file.split(",")
    if backend == "CPU":
        test_model_inference_cpu(model_file, in_data_file_list, shapes)
        print("run model inference cpu success.")
        test_parallel_inference_cpu(model_file, in_data_file_list, shapes)
        print("run parallel inference cpu success.")
    elif backend == "GPU":
        test_model_inference_gpu(model_file, in_data_file_list, shapes)
        print("run model inference gpu success.")
        test_parallel_inference_gpu(model_file, in_data_file_list, shapes)
        print("run parallel inference gpu success.")
    elif backend == "Ascend":
        test_model_inference_ascend(model_file, in_data_file_list, shapes)
        print("run model inference ascend success.")
        test_parallel_inference_ascend(model_file, in_data_file_list, shapes)
        print("run parallel inference ascend success.")
        test_input_shape_for_ascend(model_file, input_shapes_str)
        print("run input shape test success.")
    elif backend == "CPU_PARALLEL":
        test_parallel_inference_cpu(model_file, in_data_file_list, shapes)
        print("run parallel inference cpu success.")
    elif backend == "Ascend_Model_Group":
        test_model_group_for_ascend(model_file, in_data_file_list, shapes)
        print("run model model group for ascend success.")
        test_model_group_weight_workspace_for_ascend(model_file, in_data_file_list, shapes)
        print("run model group weight and workspace for ascend success.")
    else:
        raise RuntimeError('not support backend!')
    print("run success.")
