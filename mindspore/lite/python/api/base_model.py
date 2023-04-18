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
BaseModel.
"""
from mindspore_lite.tensor import Tensor


class BaseModel:
    """
    Base class for 'Model' and 'LiteInfer'.
    """
    def __init__(self, model):
        self._model = model

    def get_inputs(self):
        """
        Obtains all input Tensors of the model.
        """
        inputs = []
        for _tensor in self._model.get_inputs():
            inputs.append(Tensor(_tensor))
        return inputs

    def predict(self, inputs):
        """
        Inference model.
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be list, but got {}.".format(type(inputs)))
        _inputs = []
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
            # pylint: disable=protected-access
            _inputs.append(element._tensor)
        outputs = self._model.predict(_inputs)
        if not outputs:
            raise RuntimeError(f"predict failed!")
        predict_outputs = []
        for output in outputs:
            predict_outputs.append(Tensor(output))
        return predict_outputs

    def resize(self, inputs, dims):
        """
        Resizes the shapes of inputs.
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be list, but got {}.".format(type(inputs)))
        _inputs = []
        if not isinstance(dims, list):
            raise TypeError("dims must be list, but got {}.".format(type(dims)))
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
        for i, element in enumerate(dims):
            if not isinstance(element, list):
                raise TypeError(f"dims element must be list, but got "
                                f"{type(element)} at index {i}.")
            for j, dim in enumerate(element):
                if not isinstance(dim, int):
                    raise TypeError(f"dims element's element must be int, but got "
                                    f"{type(dim)} at {i}th dims element's {j}th element.")
        if len(inputs) != len(dims):
            raise ValueError(f"inputs' size does not match dims' size, but got "
                             f"inputs: {len(inputs)} and dims: {len(dims)}.")
        for _, element in enumerate(inputs):
            # pylint: disable=protected-access
            _inputs.append(element._tensor)
        ret = self._model.resize(_inputs, dims)
        if not ret.IsOk():
            raise RuntimeError(f"resize failed! Error is {ret.ToString()}")
