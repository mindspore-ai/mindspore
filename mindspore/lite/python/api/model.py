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
from .context import Context


class ModelType(Enum):
    MINDIR = 0
    MINDIR_LITE = 4


class Model:
    """
    The Model class is used to define a MindSpore model, facilitating computational graph management.

    Args:

    Examples:
        >>> import mindspore_lite as mslite
        >>> model = mslite.model.Model()
        >>> context.append_device_info(mslite.context.CPUDeviceInfo())
        >>> model.build_from_file("model.ms", mslite.model.ModelType.MINDIR_LITE, context)
    """

    def __init__(self):
        self._model = _c_lite_wrapper.ModelBind()
        self.model_path_ = ""

    def __str__(self):
        res = f"model_path: {self.model_path_}."
        return res

    def build_from_file(self, model_path, model_type, context):
        """
        Load and build a model from file.

        Args:
            model_path (str): Define the model path.
            model_type (ModelType): Define The type of model file.
                                              Options: ModelType::MINDIR, ModelType::MINDIR_LITE.
            context (Context): Define the context used to store options during execution.

        Raises:
            TypeError: type of input parameters are invalid.
            RuntimeError: build model failed.

        Examples:
            >>> model.build_from_file("model.ms", mslite.model.ModelType.MINDIR_LITE, context)
        """
        if not isinstance(model_path, str):
            raise TypeError("model_path must be str, but got {}.".format(type(model_path)))
        if not isinstance(model_type, ModelType):
            raise TypeError("model_type must be ModelType, but got {}.".format(type(model_type)))
        if not isinstance(context, Context):
            raise TypeError("context must be Context, but got {}.".format(type(context)))

        self.model_path_ = model_path
        model_type_ = _c_lite_wrapper.ModelType.kMindIR_Lite
        if model_type is ModelType.MINDIR:
            model_type_ = _c_lite_wrapper.ModelType.kMindIR
        ret = self._model.build_from_file(self.model_path_, model_type_, context._context)
        if ret != 0:
            raise RuntimeError(f"build_from_file failed! Error code is {ret}")

    def resize(self, inputs, dims):
        """
        Resizes the shapes of inputs.

        Args:
            inputs (list[Tensor]): A list that includes all input tensors in order.
            dims (list[list[int]]): Defines the new shapes of inputs, should be consistent with inputs.

        Raises:
            TypeError: type of input parameters are invalid.
            RuntimeError: resize model failed.

        Examples:
            >>> inputs = model.get_inputs()
            >>> model.resize(inputs, [[1, 224, 224, 3]])
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be list, but got {}.".format(type(inputs)))
        _inputs = []
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
            _inputs.append(element._tensor)
        if not isinstance(dims, list):
            raise TypeError("dims must be list, but got {}.".format(type(dims)))
        for i, element in enumerate(dims):
            if not isinstance(element, list):
                raise TypeError(f"dims element must be list, but got "
                                f"{type(element)} at index {i}.")
            for j, dim in enumerate(element):
                if not isinstance(dim, int):
                    raise TypeError(f"shape element must be int, but got "
                                    f"{type(dim)} at index {j}.")

        ret = self._model.resize(_inputs, dims)
        if ret != 0:
            raise RuntimeError(f"resize failed! Error code is {ret}")

    def predict(self, inputs, outputs):
        """
        Inference model.

        Args:
            inputs (list[Tensor]): A list that includes all input tensors in order.
            outputs (list[Tensor]): The model outputs are filled in the container in sequence.

        Raises:
            TypeError: type of input parameters are invalid.
            RuntimeError: resize model failed.

        Examples:
            >>> inputs = model.get_inputs()
            >>> outputs = model.get_outputs()
            >>> model.predict(inputs, outputs)
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be list, but got {}.".format(type(inputs)))
        if not isinstance(outputs, list):
            raise TypeError("outputs must be list, but got {}.".format(type(outputs)))
        _inputs = []
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
            _inputs.append(element._tensor)
        _outputs = []
        for i, element in enumerate(outputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"outputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
            _outputs.append(element._tensor)

        ret = self._model.predict(_inputs, _outputs, None, None)
        if ret != 0:
            raise RuntimeError(f"predict failed! Error code is {ret}")

    def get_inputs(self):
        """
        Obtains all input tensors of the model.

        Returns:
            list[Tensor], the inputs tensor list of the model.

        Examples:
            >>> inputs = model.get_inputs()
        """
        inputs = []
        for _tensor in self._model.get_inputs():
            inputs.append(Tensor(_tensor))
        return inputs

    def get_outputs(self):
        """
        Obtains all output tensors of the model.

        Returns:
            list[Tensor], the outputs tensor list of the model.

        Examples:
            >>> outputs = model.get_outputs()
        """
        outputs = []
        for _tensor in self._model.get_outputs():
            outputs.append(Tensor(_tensor))
        return outputs

    def get_input_by_tensor_name(self, tensor_name):
        """
        Obtains the input tensor of the model by name.

        Args:
            tensor_name (str): tensor name.

        Returns:
            Tensor, the input tensor of the tensor name.

        Raises:
            TypeError: type of input parameters are invalid.

        Examples:
            >>> input = model.get_input_by_tensor_name("tensor_in")
        """
        if not isinstance(tensor_name, str):
            raise TypeError("tensor_name must be str, but got {}.".format(type(tensor_name)))
        _tensor = self._model.get_input_by_tensor_name(tensor_name)
        return Tensor(_tensor)

    def get_output_by_tensor_name(self, tensor_name):
        """
        Obtains the output tensor of the model by name.

        Args:
            tensor_name (str): tensor name.

        Returns:
            Tensor, the output tensor of the tensor name.

        Raises:
            TypeError: type of input parameters are invalid.

        Examples:
            >>> output = model.get_output_by_tensor_name("tensor_out")
        """
        if not isinstance(tensor_name, str):
            raise TypeError("tensor_name must be str, but got {}.".format(type(tensor_name)))
        _tensor = self._model.get_output_by_tensor_name(tensor_name)
        return Tensor(_tensor)


class RunnerConfig:
    """
    RunnerConfig Class

    Args:
        context (Context): Define the context used to store options during execution.
        workers_num (int): the num of workers.

    Raises:
        TypeError: type of input parameters are invalid.

    Examples:
        >>> import mindspore_lite as mslite
        >>> runner_config = mslite.model.RunnerConfig(context, 4)
    """

    def __init__(self, context, workers_num):
        if not isinstance(context, Context):
            raise TypeError("context must be Context, but got {}.".format(type(context)))
        if not isinstance(workers_num, int):
            raise TypeError("workers_num must be int, but got {}.".format(type(workers_num)))
        self._runner_config = _c_lite_wrapper.RunnerConfigBind()
        self._runner_config.set_workers_num(workers_num)
        self._runner_config.set_context(context)

    def __str__(self):
        res = f"workers num: {self._runner_config.get_workers_num()}, " \
              f"context: {self._runner_config.get_context_info()}."
        return res


class ModelParallelRunner:
    """
    The ModelParallelRunner class is used to define a MindSpore ModelParallelRunner, facilitating Model management.

    Args:
        None

    Examples:
        >>> import mindspore_lite as mslite
        >>> model_parallel_runner = mslite.model.ModelParallelRunner()
    """

    def __init__(self):
        self._model = _c_lite_wrapper.ModelParallelRunnerBind()
        self.model_path_ = ""

    def __str__(self):
        return f"model_path: {self.model_path_}."

    def init(self, model_path, runner_config):
        """
        build a model parallel runner from model path so that it can run on a device.

        Args:
            model_path (str): Define the model path.
            runner_config (RunnerConfig, optional): Define the config used to store options during model pool init.

        Raises:
            TypeError: type of input parameters are invalid.
            RuntimeError: init ModelParallelRunner failed.

        Examples:
            >>> model_parallel_runner.init("test.ms", runner_config)
        """
        if not isinstance(model_path, str):
            raise TypeError("model_path must be str, but got {}.".format(type(model_path)))
        if not isinstance(runner_config, RunnerConfig):
            raise TypeError("runner_config must be RunnerConfig, but got {}.".format(type(runner_config)))

        self.model_path_ = model_path
        ret = self._model.init(self.model_path_, runner_config._runner_config)
        if ret != 0:
            raise RuntimeError(f"init failed! Error code is {ret}")

    def predict(self, inputs, outputs):
        """
        Inference ModelParallelRunner.

        Args:
            inputs (list[Tensor]): A list that includes all input tensors in order.
            outputs (list[Tensor]): The model outputs are filled in the container in sequence.

        Raises:
            TypeError: type of input parameters are invalid.
            RuntimeError: resize model failed.

        Examples:
            >>> inputs = model_parallel_runner.get_inputs()
            >>> outputs = model_parallel_runner.get_outputs()
            >>> model_parallel_runner.predict(inputs, outputs)
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be list, but got {}.".format(type(inputs)))
        if not isinstance(outputs, list):
            raise TypeError("outputs must be list, but got {}.".format(type(outputs)))
        _inputs = []
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
            _inputs.append(element._tensor)
        _outputs = []
        for i, element in enumerate(outputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"outputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
            _outputs.append(element._tensor)

        ret = self._model.predict(_inputs, _outputs, None, None)
        if ret != 0:
            raise RuntimeError(f"predict failed! Error code is {ret}")

    def get_inputs(self):
        """
        Obtains all input tensors of the model.

        Returns:
            list[Tensor], the inputs tensor list of the model.

        Examples:
            >>> inputs = model_parallel_runner.get_inputs()
        """
        inputs = []
        for _tensor in self._model.get_inputs():
            inputs.append(Tensor(_tensor))
        return inputs

    def get_outputs(self):
        """
        Obtains all output tensors of the model.

        Returns:
            list[Tensor], the outputs tensor list of the model.

        Examples:
            >>> outputs = model_parallel_runner.get_outputs()
        """
        outputs = []
        for _tensor in self._model.get_outputs():
            outputs.append(Tensor(_tensor))
        return outputs
