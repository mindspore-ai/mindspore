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
from mindspore_lite.tensor import Tensor, TensorMeta


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

    def get_outputs(self):
        """
        Obtains all output TensorMeta of the model.
        """
        outputs_metadata = []
        for _tensor in self._model.get_outputs():
            out_tensor = Tensor(_tensor)
            output_meta = TensorMeta()
            output_meta.name = out_tensor.name
            output_meta.dtype = out_tensor.dtype
            output_meta.shape = out_tensor.shape
            output_meta.format = out_tensor.format
            output_meta.element_num = out_tensor.element_num
            output_meta.data_size = out_tensor.data_size
            outputs_metadata.append(output_meta)
        return tuple(outputs_metadata)

    def get_model_info(self, key):
        """
        Obtains model info of the model.

        Args:
            key (str): Get model information keywords, currently user_info, input_shape,
                dynamic_dims, user_info indicate user information, input_shape indicates input shape for the model,
                and dynamic_dims is the binning supported by the model.

        Raises:
            TypeError: If the key is not a str.
        """
        if not isinstance(key, str):
            raise TypeError("key must be str, but got {}.".format(type(key)))
        return self._model.get_model_info(key)

    def update_weights(self, weights):
        """
        Update constant weight of the model node.
        """
        if not isinstance(weights, list):
            raise TypeError("weights must be list, but got {}.".format(type(weights)))
        _weights = []
        for i, weight in enumerate(weights):
            _weight = []
            if not isinstance(weight, list):
                raise TypeError("weight must be list, but got {}.".format(type(weight)))
            # pylint: disable=protected-access
            for j, tensor in enumerate(weight):
                if not isinstance(tensor, Tensor):
                    raise TypeError(f"weights element must be Tensor, but got "
                                    f"{type(tensor)} at index {i}{j}.")
                _weight.append(tensor._tensor)
            _weights.append(_weight)
        ret = self._model.update_weights(_weights)
        if not ret.IsOk():
            raise RuntimeError(f"update weight failed! Error is {ret.ToString()}")

    def predict(self, inputs, outputs=None):
        """
        Inference model.
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be list, but got {}.".format(type(inputs)))
        _inputs = []
        _outputs = []
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
            # pylint: disable=protected-access
            _inputs.append(element._tensor)
        if outputs is not None:
            if not isinstance(outputs, list):
                raise TypeError("inputs must be list, but got {}.".format(type(inputs)))
            for i, element in enumerate(outputs):
                if not isinstance(element, Tensor):
                    raise TypeError(f"outputs element must be Tensor, but got "
                                    f"{type(element)} at index {i}.")
                # pylint: disable=protected-access
                _outputs.append(element._tensor)
        predict_result = self._model.predict(_inputs, _outputs)
        if predict_result is None or len(predict_result) == 0:
            raise RuntimeError(f"predict failed!")
        predict_outputs = []
        for output_tensor in predict_result:
            predict_outputs.append(Tensor(output_tensor))
        return predict_outputs

    def resize(self, inputs, dims):
        """
        Resizes the shapes of inputs.
        """
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("inputs must be list or tuple of Lite Tensor, but got {}.".format(type(inputs)))
        _inputs = []
        if not isinstance(dims, (list, tuple)):
            raise TypeError("dims must be list or tuple of shape, but got {}.".format(type(dims)))
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
        for i, element in enumerate(dims):
            if not isinstance(element, (list, tuple)):
                raise TypeError(f"dims element must be list or tuple of int, but got "
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
