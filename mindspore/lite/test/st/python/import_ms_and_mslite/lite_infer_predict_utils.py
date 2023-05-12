# Copyright 2023 Huawei Technologies Co., Ltd
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
LIte Infer model.predict util.
"""

import time

import numpy as np

import mindspore as ms


def predict_backend_lite(ms_model, data_input):
    """
    model.predict using backend lite.
    """
    # model predict using backend lite
    output = ms_model.predict(data_input, backend="lite")
    t_start = time.time()
    for _ in range(100):
        ms_model.predict(data_input, backend="lite")
    avg_time = (time.time() - t_start) / 100
    return output, avg_time


def predict_mindir(ms_model, data_input):
    """
    predict by MindIR.
    """
    def _predict_core(lite_mode_input):
        """single input."""
        inputs = lite_mode_input.get_inputs()
        if len(inputs) > 1:
            raise RuntimeError("Only support single input in this net.")
        inputs[0].set_data_from_numpy(data_input.asnumpy())
        outputs = lite_mode_input.predict(inputs)
        return ms.Tensor(outputs[0].get_data_to_numpy())

    def _get_lite_context(l_context):
        lite_context_properties = {
            "cpu": ["inter_op_parallel_num", "precision_mode", "thread_num",
                    "thread_affinity_mode", "thread_affinity_core_list"],
            "gpu": ["device_id", "precision_mode"],
            "ascend": ["device_id", "precision_mode", "provider", "rank_id"]
        }
        lite_device_target = ms.get_context('device_target').lower()
        if lite_device_target not in ['cpu', 'gpu', 'ascend']:
            raise RuntimeError(f"Device target should be in ['cpu', 'gpu', 'ascend'], but got {lite_device_target}")
        l_context.target = [lite_device_target]
        l_context_device_dict = {'cpu': l_context.cpu, 'gpu': l_context.gpu, 'ascend': l_context.ascend}
        for single_property in lite_context_properties.get(lite_device_target):
            try:
                context_value = ms.get_context(single_property)
                if context_value:
                    setattr(l_context_device_dict.get(lite_device_target), single_property, context_value)
            except ValueError:
                print(f'For set lite context, fail to get parameter {single_property} from ms.context.'
                      f' Will use default value')
        return l_context

    try:
        import mindspore_lite as mslite
    except ImportError:
        raise ImportError(f"For predict by MindIR, mindspore_lite should be installed.")

    lite_context = mslite.Context()
    lite_context = _get_lite_context(lite_context)

    ms.export(ms_model.predict_network, data_input, file_name="net", file_format="MINDIR")

    lite_model = mslite.Model()
    lite_model.build_from_file("net.mindir", mslite.ModelType.MINDIR, lite_context)

    output = _predict_core(lite_model)
    t_start = time.time()
    for _ in range(100):
        _predict_core(lite_model)
    avg_time = (time.time() - t_start) / 100
    return output, avg_time


def predict_lenet(ms_model, data_input):
    """
    ms.Model.predict
    """
    # model predict
    output = ms_model.predict(data_input)
    t_start = time.time()
    for _ in range(100):
        ms_model.predict(data_input)
    avg_time = (time.time() - t_start) / 100
    return output, avg_time


def _get_max_index_from_res(data_input):
    data_input = data_input.asnumpy()
    data_input = data_input.flatten()
    res_index = np.where(data_input == np.max(data_input))  # (array([6]), )
    return res_index[0][0]
