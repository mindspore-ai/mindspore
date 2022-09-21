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
import os
from enum import Enum

from ._checkparam import check_isinstance
from .context import Context
from .lib import _c_lite_wrapper
from .tensor import Tensor

__all__ = ['ModelType', 'Model', 'RunnerConfig', 'ModelParallelRunner']


class ModelType(Enum):
    """
    The MoedelType is used to define the model type.
    """
    MINDIR = 0
    MINDIR_LITE = 4


model_type_py_cxx_map = {
    ModelType.MINDIR: _c_lite_wrapper.ModelType.kMindIR,
    ModelType.MINDIR_LITE: _c_lite_wrapper.ModelType.kMindIR_Lite,
}

model_type_cxx_py_map = {
    _c_lite_wrapper.ModelType.kMindIR: ModelType.MINDIR,
    _c_lite_wrapper.ModelType.kMindIR_Lite: ModelType.MINDIR_LITE,
}


class Model:
    """
    The Model class is used to define a MindSpore model, facilitating computational graph management.

    Examples:
        >>> import mindspore_lite as mslite
        >>> model = mslite.Model()
        >>> print(model)
        model_path: .
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
            model_path (str): Define the model path, include model file.
            model_type (ModelType): Define The type of model file. Options: ModelType::MINDIR | ModelType::MINDIR_LITE.

                - ModelType::MINDIR: An intermediate representation of the MindSpore model.
                  The recommended model file suffix is ".mindir".
                - ModelType::MINDIR_LITE: An intermediate representation of the MindSpore Lite model.
                  The recommended model file suffix is ".ms".

            context (Context): Define the context used to store options during execution.

        Raises:
            TypeError: `model_path` is not a str.
            TypeError: `model_type` is not a ModelType.
            TypeError: `context` is not a Context.
            RuntimeError: `model_path` does not exist.

        Examples:
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE, context)
            >>> print(model)
            model_path: mobilenetv2.ms.
        """
        check_isinstance("model_path", model_path, str)
        check_isinstance("model_type", model_type, ModelType)
        check_isinstance("context", context, Context)
        if not os.path.exists(model_path):
            raise RuntimeError(f"build_from_file failed, model_path does not exist!")
        self.model_path_ = model_path
        model_type_ = _c_lite_wrapper.ModelType.kMindIR_Lite
        if model_type is ModelType.MINDIR:
            model_type_ = _c_lite_wrapper.ModelType.kMindIR
        ret = self._model.build_from_file(self.model_path_, model_type_, context._context)
        if not ret.IsOk():
            raise RuntimeError(f"build_from_file failed! Error is {ret.ToString()}")

    def resize(self, inputs, dims):
        """
        Resizes the shapes of inputs.

        Args:
            inputs (list[Tensor]): A list that includes all input tensors in order.
            dims (list[list[int]]): A list that includes the new shapes of inputs, should be consistent with inputs.

        Raises:
            TypeError: `inputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            TypeError: `dims` is not a list.
            TypeError: `dims` is a list, but the elements are not list.
            TypeError: `dims` is a list, the elements are list, but the element's elements are not int.
            ValueError: The size of `inputs` is not equal to the size of `dims` .
            ValueError: The size of the elements of `inputs` is not equal to the size of the elements of `dims` .

        Examples:
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE, context)
            >>> inputs = model.get_inputs()
            >>> print("Before resize, the first input shape: ", inputs[0].get_shape())
            Before resize, the first input shape: [1, 224, 224, 3]
            >>> model.resize(inputs, [[1, 112, 112, 3]])
            >>> print("After resize, the first input shape: ", inputs[0].get_shape())
            After resize, the first input shape: [1, 112, 112, 3]
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
            raise ValueError(f"inputs' size does not match dims's size, but got "
                             f"inputs: {len(inputs)} and dims: {len(dims)}.")
        for i, element in enumerate(inputs):
            if len(element.get_shape()) != len(dims[i]):
                raise ValueError(f"one of inputs' size does not match one of dims's size, but got "
                                 f"input: {element.get_shape()} and dim: {len(dims[i])} at {i} index.")
            _inputs.append(element._tensor)
        ret = self._model.resize(_inputs, dims)
        if not ret.IsOk():
            raise RuntimeError(f"resize failed! Error is {ret.ToString()}")

    def predict(self, inputs, outputs):
        """
        Inference model.

        Args:
            inputs (list[Tensor]): A list that includes all input tensors in order.
            outputs (list[Tensor]): The model outputs are filled in the container in sequence.

        Raises:
            TypeError: `inputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            TypeError: `outputs` is not a list.
            TypeError: `outputs` is a list, but the elements are not Tensor.
            RuntimeError: predict model failed.

        Examples:
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> # in_data download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/input.bin
            >>> # 1. predict which indata is from file
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE, context)
            >>> inputs = model.get_inputs()
            >>> outputs = model.get_outputs()
            >>> in_data = np.fromfile("input.bin", dtype=np.float32)
            >>> inputs[0].set_data_from_numpy(in_data)
            >>> model.predict(inputs, outputs)
            >>> for output in outputs:
            ...     data = output.get_data_to_numpy()
            ...     print("outputs: ", data)
            ...
            outputs:  [[1.02271215e-05 9.92699006e-06 1.69684317e-05 ... 6.69087376e-06
                        2.16263197e-06 1.24009384e-04]]
            >>> # 2. predict which indata is numpy array
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE, context)
            >>> inputs = model.get_inputs()
            >>> outputs = model.get_outputs()
            >>> for input in inputs:
            ...     in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
            ...     input.set_data_from_numpy(in_data)
            ...
            >>> model.predict(inputs, outputs)
            >>> for output in outputs:
            ...     data = output.get_data_to_numpy()
            ...     print("outputs: ", data)
            ...
            outputs:  [[0.00035889 0.00065501 0.00052925 ... 0.00018388 0.00148316 0.00116824]]
            >>> # 3. predict which indata is new mslite tensor with numpy array
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE, context)
            >>> inputs = model.get_inputs()
            >>> outputs = model.get_outputs()
            >>> input_tensors = []
            >>> for input in inputs:
            ...     input_tensor = mslite.Tensor()
            ...     input_tensor.set_data_type(input.get_data_type())
            ...     input_tensor.set_shape(input.get_shape())
            ...     input_tensor.set_format(input.get_format())
            ...     input_tensor.set_tensor_name(input.get_tensor_name())
            ...     in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
            ...     input_tensor.set_data_from_numpy(in_data)
            ...     input_tensors.append(input_tensor)
            ...
            >>> model.predict(input_tensors, outputs)
            >>> for output in outputs:
            ...     data = output.get_data_to_numpy()
            ...     print("outputs: ", data)
            ...
            outputs:  [[0.00035889 0.00065501 0.00052925 ... 0.00018388 0.00148316 0.00116824]]
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
        if not ret.IsOk():
            raise RuntimeError(f"predict failed! Error is {ret.ToString()}")

    def get_inputs(self):
        """
        Obtains all input tensors of the model.

        Returns:
            list[Tensor], the inputs tensor list of the model.

        Examples:
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE, context)
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
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE, context)
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
            TypeError: `tensor_name` is not a str.
            RuntimeError: get input by tensor name failed.

        Examples:
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE, context)
            >>> input_tensor = model.get_input_by_tensor_name("graph_input-173")
            >>> print(input_tensor)
            tensor_name: graph_input-173,
            data_type: DataType.FLOAT32,
            shape: [1, 224, 224, 3],
            format: Format.NHWC,
            element_num: 150528,
            data_size: 602112.
        """
        check_isinstance("tensor_name", tensor_name, str)
        _tensor = self._model.get_input_by_tensor_name(tensor_name)
        if _tensor.is_null():
            raise RuntimeError(f"get_input_by_tensor_name failed!")
        return Tensor(_tensor)

    def get_output_by_tensor_name(self, tensor_name):
        """
        Obtains the output tensor of the model by name.

        Args:
            tensor_name (str): tensor name.

        Returns:
            Tensor, the output tensor of the tensor name.

        Raises:
            TypeError: `tensor_name` is not a str.
            RuntimeError: get output by tensor name failed.

        Examples:
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE, context)
            >>> output_tensor = model.get_output_by_tensor_name("Softmax-65")
            >>> print(output_tensor)
            tensor_name: Softmax-65,
            data_type: DataType.FLOAT32,
            shape: [1, 1001],
            format: Format.NHWC,
            element_num: 1001,
            data_size: 4004.
        """
        check_isinstance("tensor_name", tensor_name, str)
        _tensor = self._model.get_output_by_tensor_name(tensor_name)
        if _tensor.is_null():
            raise RuntimeError(f"get_output_by_tensor_name failed!")
        return Tensor(_tensor)


class RunnerConfig:
    """
    RunnerConfig Class defines runner config of one or more servables.
    The class can be used to make model parallel runner which corresponds to the service provided by a model.
    The client sends inference tasks and receives inference results through server.

    Args:
        context (Context, optional): Define the context used to store options during execution. Default: None.
        workers_num (int, optional): the num of workers. Default: None.
        config_info (dict{str, dict{str, str}}, optional): Nested map for passing model weight paths.
            e.g. {"weight": {"weight_path": "/home/user/weight.cfg"}}. Default: None.
            key currently supports ["weight"];
            value is in dict format, key of it currently supports ["weight_path"],
            value of it is the path of weight, e.g. "/home/user/weight.cfg".

    Raises:
        TypeError: `context` is neither a Context nor None.
        TypeError: `workers_num` is neither an int nor None.
        TypeError: `config_info` is neither a dict nor None.
        TypeError: `config_info` is a dict, but the key is not str.
        TypeError: `config_info` is a dict, the key is str, but the value is not dict.
        TypeError: `config_info` is a dict, the key is str, the value is dict, but the key of value is not str.
        TypeError: `config_info` is a dict, the key is str, the value is dict, the key of the value is str, but
            the value of the value is not str.
        ValueError: `workers_num` is an int, but it is less than 0.

    Examples:
        >>> # only for serving inference
        >>> import mindspore_lite as mslite
        >>> context = mslite.Context()
        >>> context.append_device_info(mslite.CPUDeviceInfo())
        >>> config_info = {"weight": {"weight_path": "path of model weight"}}
        >>> runner_config = mslite.RunnerConfig(context=context, workers_num=0, config_info=config_info)
        >>> print(runner_config)
        workers num: 0,
        config info: weight: weight_path: path of model weight
        ,
        context: thread num: 0, bind mode: 1.
    """

    def __init__(self, context=None, workers_num=None, config_info=None, config_path=None):
        if context is not None:
            check_isinstance("context", context, Context)
        if workers_num is not None:
            check_isinstance("workers_num", workers_num, int)
            if workers_num < 0:
                raise ValueError(f"RunnerConfig's init failed! workers_num must be positive.")
        if config_info is not None:
            check_isinstance("config_info", config_info, dict)
            for k, v in config_info.items():
                check_isinstance("config_info_key", k, str)
                check_isinstance("config_info_value", v, dict)
                for v_k, v_v in v.items():
                    check_isinstance("config_info_value_key", v_k, str)
                    check_isinstance("config_info_value_value", v_v, str)
        self._runner_config = _c_lite_wrapper.RunnerConfigBind()
        if context is not None:
            self._runner_config.set_context(context._context)
        if workers_num is not None:
            self._runner_config.set_workers_num(workers_num)
        if config_info is not None:
            for k, v in config_info.items():
                self._runner_config.set_config_info(k, v)
        if config_path is not None:
            if not os.path.exists(config_path):
                raise ValueError(f"RunnerConfig's init failed, config_path does not exist!")
            check_isinstance("config_path", config_path, str)
            self._runner_config.set_config_path(config_path)

    def __str__(self):
        res = f"workers num: {self._runner_config.get_workers_num()},\n" \
              f"config info: {self._runner_config.get_config_info_string()},\n" \
              f"context: {self._runner_config.get_context_info()},\n" \
              f"config path: {self._runner_config.get_config_path()}."
        return res


class ModelParallelRunner:
    """
    The ModelParallelRunner class is used to define a MindSpore ModelParallelRunner, facilitating Model management.

    Examples:
        >>> # only for serving inference
        >>> import mindspore_lite as mslite
        >>> model_parallel_runner = mslite.ModelParallelRunner()
        >>> print(model_parallel_runner)
        model_path: .
    """

    def __init__(self):
        self._model = _c_lite_wrapper.ModelParallelRunnerBind()
        self.model_path_ = ""

    def __str__(self):
        return f"model_path: {self.model_path_}."

    def init(self, model_path, runner_config=None):
        """
        build a model parallel runner from model path so that it can run on a device.

        Args:
            model_path (str): Define the model path.
            runner_config (RunnerConfig, optional): Define the config used to store options during model pool init.
                Default: None.

        Raises:
            TypeError: `model_path` is not a str.
            TypeError: `runner_config` is neither a RunnerConfig nor None.
            RuntimeError: `model_path` does not exist.
            RuntimeError: ModelParallelRunner's init failed.

        Examples:
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> runner_config = mslite.RunnerConfig(context=context, workers_num=4)
            >>> model_parallel_runner = mslite.ModelParallelRunner()
            >>> model_parallel_runner.init(model_path="mobilenetv2.ms", runner_config=runner_config)
            >>> print(model_parallel_runner)
            model_path: mobilenetv2.ms.
        """
        check_isinstance("model_path", model_path, str)
        if not os.path.exists(model_path):
            raise RuntimeError(f"ModelParallelRunner's init failed, model_path does not exist!")
        self.model_path_ = model_path
        if runner_config is not None:
            check_isinstance("runner_config", runner_config, RunnerConfig)
            ret = self._model.init(self.model_path_, runner_config._runner_config)
        else:
            ret = self._model.init(self.model_path_, None)
        if not ret.IsOk():
            raise RuntimeError(f"ModelParallelRunner's init failed! Error is {ret.ToString()}")

    def predict(self, inputs, outputs):
        """
        Inference ModelParallelRunner.

        Args:
            inputs (list[Tensor]): A list that includes all input tensors in order.
            outputs (list[Tensor]): The model outputs are filled in the container in sequence.

        Raises:
            TypeError: `inputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            TypeError: `outputs` is not a list.
            TypeError: `outputs` is a list, but the elements are not Tensor.
            RuntimeError: predict model failed.

        Examples:
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> # in_data download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/input.bin
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> runner_config = mslite.RunnerConfig(context=context, workers_num=4)
            >>> model_parallel_runner = mslite.ModelParallelRunner()
            >>> model_parallel_runner.init(model_path="mobilenetv2.ms", runner_config=runner_config)
            >>> inputs = model_parallel_runner.get_inputs()
            >>> in_data = np.fromfile("input.bin", dtype=np.float32)
            >>> inputs[0].set_data_from_numpy(in_data)
            >>> outputs = model_parallel_runner.get_outputs()
            >>> model_parallel_runner.predict(inputs, outputs)
            >>> for output in outputs:
            ...     data = output.get_data_to_numpy()
            ...     print("outputs: ", data)
            ...
            outputs:  [[1.02271215e-05 9.92699006e-06 1.69684317e-05 ... 6.69087376e-06
                        2.16263197e-06 1.24009384e-04]]
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
        predict_output = self._model.predict(_inputs, _outputs, None, None)
        if not isinstance(predict_output, list) or len(predict_output) == 0:
            raise RuntimeError(f"predict failed!")
        outputs.clear()
        for i, element in enumerate(predict_output):
            outputs.append(Tensor(element))

    def get_inputs(self):
        """
        Obtains all input tensors of the model.

        Returns:
            list[Tensor], the inputs tensor list of the model.

        Examples:
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> runner_config = mslite.RunnerConfig(context=context, workers_num=4)
            >>> model_parallel_runner = mslite.ModelParallelRunner()
            >>> model_parallel_runner.init(model_path="mobilenetv2.ms", runner_config=runner_config)
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
            >>> # model download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> context.append_device_info(mslite.CPUDeviceInfo())
            >>> runner_config = mslite.RunnerConfig(context=context, workers_num=4)
            >>> model_parallel_runner = mslite.ModelParallelRunner()
            >>> model_parallel_runner.init(model_path="mobilenetv2.ms", runner_config=runner_config)
            >>> outputs = model_parallel_runner.get_outputs()
        """
        outputs = []
        for _tensor in self._model.get_outputs():
            outputs.append(Tensor(_tensor))
        return outputs
