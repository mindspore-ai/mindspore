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
Model API.
"""
from enum import Enum
from .lib import _c_lite_wrapper
from .tensor import Tensor


class ModelType(Enum):
    MINDIR = 0
    MINDIR_LITE = 4


class Model:
    """
    Model Class

    Args:
    """
    def __init__(self):
        self._model = _c_lite_wrapper.ModelBind()

    def build_from_file(self, model_path, model_type, context):
        _model_type = _c_lite_wrapper.ModelType.kMindIR_Lite
        if model_type is ModelType.MINDIR:
            _model_type = _c_lite_wrapper.ModelType.kMindIR
        return self._model.build_from_file(model_path, _model_type, context._context)

    def resize(self, inputs, dims):
        return self._model.resize(inputs, dims)

    def predict(self, inputs, outputs, before=None, after=None):
        """model predict"""
        _inputs = []
        for tensor in inputs:
            _inputs.append(tensor._tensor)
        _outputs = []
        for tensor in outputs:
            _outputs.append(tensor._tensor)
        ret = self._model.predict(_inputs, _outputs, before, after)
        if ret != 0:
            raise RuntimeError(f"Predict failed! Error code is {ret}")
        return ret

    def get_inputs(self):
        inputs = []
        for _tensor in self._model.get_inputs():
            inputs.append(Tensor(_tensor))
        return inputs

    def get_outputs(self):
        outputs = []
        for _tensor in self._model.get_outputs():
            outputs.append(Tensor(_tensor))
        return outputs

    def get_input_by_tensor_name(self, tensor_name):
        return self._model.get_input_by_tensor_name(tensor_name)

    def get_output_by_tensor_name(self, tensor_name):
        return self._model.get_output_by_tensor_name(tensor_name)


class ModelParallelRunner:
    """
    ModelParallelRunner Class

    Args:
    """
    def __init__(self, model_path, context, workers_num):
        self._model = _c_lite_wrapper.ModelParallelRunnerBind(model_path, context._context, workers_num)

    def predict(self, inputs, outputs, before=None, after=None):
        """model predict"""
        _inputs = []
        for tensor in inputs:
            _inputs.append(tensor._tensor)
        _outputs = []
        for tensor in outputs:
            _outputs.append(tensor._tensor)
        ret = self._model.predict(_inputs, _outputs, before, after)
        if ret != 0:
            raise RuntimeError(f"Predict failed! Error code is {ret}")
        return ret

    def get_inputs(self):
        inputs = []
        for _tensor in self._model.get_inputs():
            inputs.append(Tensor(_tensor))
        return inputs

    def get_outputs(self):
        outputs = []
        for _tensor in self._model.get_outputs():
            outputs.append(Tensor(_tensor))
        return outputs
