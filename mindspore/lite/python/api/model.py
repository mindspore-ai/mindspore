# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import
import os
import logging
from enum import Enum
import numpy as np

from mindspore_lite._checkparam import check_isinstance
from mindspore_lite.context import Context
from mindspore_lite.lib import _c_lite_wrapper
from mindspore_lite.tensor import Tensor
from mindspore_lite.base_model import BaseModel
from mindspore_lite._parse_update_weights_name import _parse_update_weight_config_name, _rename_variable_weight

__all__ = ['ModelType', 'Model', 'ModelParallelRunner', 'ModelGroup']


class ModelType(Enum):
    """
    The `ModelType` class defines the type of the model exported or imported in MindSpot Lite.

    Used in the following scenarios:

    1. When using `mindspore_lite.Converter`, set `save_type` parameter, `ModelType` used to define the model type
    generated by Converter. ``ModelType.MINDIR`` is recommended.

    2. After using `mindspore_lite.Converter`, when loading or building a model from file for predicting, the
    `ModelType` is used to define Input model framework type. Only support ``ModelType.MINDIR``.

    Currently, the following `ModelType` are supported:

    ===========================  =======================================================================
    Definition                    Description
    ===========================  =======================================================================
    `ModelType.MINDIR`           MindSpore model's framework type, which model uses .mindir as suffix.
    `ModelType.MINDIR_LITE`      MindSpore Lite model's framework type, which model uses .ms as suffix.
    ===========================  =======================================================================

    Examples:
        >>> # Method 1: Import mindspore_lite package
        >>> import mindspore_lite as mslite
        >>> print(mslite.ModelType.MINDIR)
        ModelType.MINDIR
        >>> # Method 2: from mindspore_lite package import ModelType
        >>> from mindspore_lite import ModelType
        >>> print(ModelType.MINDIR)
        ModelType.MINDIR
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


def set_env(func):
    """set env for Ascend custom opp"""

    def wrapper(*args, **kwargs):
        current_path = os.path.dirname(os.path.abspath(__file__))
        mslite_ascend_ascendc_custom_kernel_path = os.path.join(current_path,
                                                                "custom_kernels",
                                                                "ascend", "ascendc",
                                                                "packages", "vendors", "mslite_ascendc")
        mslite_ascend_tbe_custom_kernel_path = os.path.join(current_path,
                                                            "custom_kernels",
                                                            "ascend", "tbe_and_aicpu", "packages",
                                                            "vendors", "mslite_tbe_and_aicpu")
        if os.path.exists(mslite_ascend_tbe_custom_kernel_path):
            if os.getenv('ASCEND_CUSTOM_OPP_PATH'):
                os.environ['ASCEND_CUSTOM_OPP_PATH'] = mslite_ascend_tbe_custom_kernel_path + ":" + \
                                                       os.environ['ASCEND_CUSTOM_OPP_PATH']
            else:
                os.environ['ASCEND_CUSTOM_OPP_PATH'] = mslite_ascend_tbe_custom_kernel_path
        else:
            logging.warning(
                "mslite tbe_and_aicpu custom kernel path not found")
        if os.path.exists(mslite_ascend_ascendc_custom_kernel_path):
            if os.getenv('ASCEND_CUSTOM_OPP_PATH'):
                os.environ['ASCEND_CUSTOM_OPP_PATH'] = mslite_ascend_ascendc_custom_kernel_path + ":" + \
                                                       os.environ['ASCEND_CUSTOM_OPP_PATH']
            else:
                os.environ['ASCEND_CUSTOM_OPP_PATH'] = mslite_ascend_ascendc_custom_kernel_path
        else:
            logging.warning("mslite ascendc custom kernel path not found")
        return func(*args, **kwargs)

    return wrapper


class Model(BaseModel):
    """
    The `Model` class defines a MindSpore Lite's model, facilitating computational graph management.

    Examples:
        >>> import mindspore_lite as mslite
        >>> model = mslite.Model()
        >>> print(model)
        model_path: .
    """

    def __init__(self):
        super(Model, self).__init__(_c_lite_wrapper.ModelBind())
        self.model_path_ = ""

    def __str__(self):
        res = f"model_path: {self.model_path_}."
        return res

    @set_env
    def build_from_file(self, model_path, model_type, context=None, config_path="", config_dict: dict = None):
        """
        Load and build a model from file.

        Args:
            model_path (str): Path of the input model when build from file. For example, "/home/user/model.mindir".
                Model should use .mindir as suffix.
            model_type (ModelType): Define The type of input model file. Option is ``ModelType.MINDIR``.
                For details, see
                `ModelType <https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.ModelType.html>`_ .
            context (Context, optional): Define the context used to transfer options during execution.
                Default: ``None``. ``None`` means the Context with cpu target.
            config_path (str, optional): Define the config file path. the config file is used to transfer user defined
                options during build model. In the following scenarios, users may need to set the parameter.
                For example, "/home/user/config.txt". Default: ``""``.

                - Usage 1: Set mixed precision inference. The content and description of the configuration file are as
                  follows:

                  .. code-block::

                      [execution_plan]
                      [op_name1]=data_Type: float16 (The operator named op_name1 sets the data type as float16)
                      [op_name2]=data_Type: float32 (The operator named op_name2 sets the data type as float32)

                - Usage 2: When GPU inference, set the configuration of TensorRT. The content and description of the
                  configuration file are as follows:

                  .. code-block::

                      [ms_cache]
                      serialize_Path=[serialization model path](storage path of serialization model)
                      [gpu_context]
                      input_shape=input_Name: [input_dim] (Model input dimension, for dynamic shape)
                      dynamic_Dims=[min_dim~max_dim] (dynamic dimension range of model input, for dynamic shape)
                      opt_Dims=[opt_dim] (the optimal input dimension of the model, for dynamic shape)

            config_dict (dict, optional): When you set config in this dict, the priority is higher than the
                configuration items in config_path.

                Set rank table file for inference. The content of the configuration file is as follows:

                .. code-block::

                    [ascend_context]
                    rank_table_file=[path_a](storage initial path of the rank table file)

                When set

                .. code-block::

                    config_dict = {"ascend_context" : {"rank_table_file" : "path_b"}}

                The the path_b from the config_dict will be used to compile the model.

        Raises:
            TypeError: `model_path` is not a str.
            TypeError: `model_type` is not a ModelType.
            TypeError: `context` is neither a Context nor ``None``.
            TypeError: `config_path` is not a str.
            RuntimeError: `model_path` does not exist.
            RuntimeError: `config_path` does not exist.
            RuntimeError: load configuration by `config_path` failed.
            RuntimeError: build from file failed.

        Examples:
            >>> # Testcase 1: build from file with default cpu context.
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> model.build_from_file("mobilenetv2.mindir", mslite.ModelType.MINDIR)
            >>> print(model)
            model_path: mobilenetv2.mindir.
            >>> # Testcase 2: build from file with gpu context.
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.target = ["cpu"]
            >>> model.build_from_file("mobilenetv2.mindir", mslite.ModelType.MINDIR, context)
            >>> print(model)
            model_path: mobilenetv2.mindir.
        """
        check_isinstance("model_path", model_path, str)
        check_isinstance("model_type", model_type, ModelType)
        if context is None:
            context = Context()
        check_isinstance("context", context, Context)
        check_isinstance("config_path", config_path, str)
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"build_from_file failed, model_path does not exist!")
        self.model_path_ = model_path
        model_type_ = _c_lite_wrapper.ModelType.kMindIR_Lite
        if model_type is ModelType.MINDIR:
            model_type_ = _c_lite_wrapper.ModelType.kMindIR
        if config_path:
            if not os.path.exists(config_path):
                raise RuntimeError(
                    f"build_from_file failed, config_path does not exist!")
            ret = self._model.load_config(config_path)
            if not ret.IsOk():
                raise RuntimeError(
                    f"load configuration failed! Error is {ret.ToString()}")
            update_names = _parse_update_weight_config_name(config_path)
            if update_names is not None:
                if config_dict is None:
                    config_dict = {"ascend_context": {"variable_weights_list": update_names}}
                else:
                    config_dict['ascend_context']["variable_weights_list"] = update_names

        if config_dict:
            check_isinstance("config_dict", config_dict, dict)
            for k, v in config_dict.items():
                check_isinstance("config_dict_key", k, str)
                check_isinstance("config_dict_value", v, dict)
                for v_k, v_v in v.items():
                    check_isinstance("config_dict_value_key", v_k, str)
                    check_isinstance("config_dict_value_value", v_v, str)
            for key, value in config_dict.items():
                ret = self._model.update_config(key, value)
                if not ret.IsOk():
                    raise RuntimeError(f"update configuration failed! Error is {ret.ToString()}."
                                       f"Setcion is {key}, config is {value}")

        ret = self._model.build_from_file(
            self.model_path_, model_type_, context._context._inner_context)
        if not ret.IsOk():
            raise RuntimeError(
                f"build_from_file failed! Error is {ret.ToString()}")

    def get_outputs(self):
        """
        Obtains all output information Tensors of the model.

        Returns:
            list[TensorMeta], the output TensorMeta list of the model.
        """
        # pylint: disable=useless-super-delegation
        return super(Model, self).get_outputs()

    def get_inputs(self):
        """
        Obtains all input Tensors of the model.

        Returns:
            list[Tensor], the input Tensor list of the model.

        Examples:
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> model.build_from_file("mobilenetv2.mindir", mslite.ModelType.MINDIR)
            >>> inputs = model.get_inputs()
        """
        # pylint: disable=useless-super-delegation
        return super(Model, self).get_inputs()

    def update_weights(self, weights):
        """
        Update constant weight of the model node.

        Args:
            weights (list[list[Tensor]]): A list that includes all update weight Tensors.

        Raises:
            RuntimeError: `weights` is not a list(list).
            RuntimeError: `weights` is a list, but the elements are not Tensor.
            RuntimeError: update weight failed.

        Tutorial Examples:
            - `Dynamic Weight Update
              <https://www.mindspore.cn/lite/docs/en/r2.3/use/cloud_infer/runtime_python.html#dynamic-weight-update>`_
        """
        for weight in weights:
            for tensor in weight:
                name = _rename_variable_weight(tensor.name)
                tensor.name = name
        return super(Model, self).update_weights(weights)

    def predict(self, inputs, outputs=None):
        """
        Inference model.

        Args:
            inputs (list[Tensor]): A list that includes all input Tensors in order.
            outputs (list[Tensor], optional): A list that includes all output Tensors in order,
                this tensor include output data buffer.

        Returns:
            list[Tensor], the output Tensor list of the model.

        Raises:
            TypeError: `inputs` is not a list.
            TypeError: `outputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            TypeError: `outputs` is a list, but the elements are not Tensor.
            RuntimeError: predict model failed.

        Examples:
            >>> # 1. predict which indata is from file
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> model = mslite.Model()
            >>> #default context's target is cpu
            >>> model.build_from_file("mobilenetv2.mindir", mslite.ModelType.MINDIR)
            >>> inputs = model.get_inputs()
            >>> in_data = np.fromfile("input.bin", dtype=np.float32)
            >>> inputs[0].set_data_from_numpy(in_data)
            >>> outputs = model.predict(inputs)
            >>> for output in outputs:
            ...     data = output.get_data_to_numpy()
            ...     print("outputs' shape: ", data.shape)
            ...
            outputs' shape:  (1,1001)
            >>> # 2. predict which indata is numpy array
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> model = mslite.Model()
            >>> model.build_from_file("mobilenetv2.mindir", mslite.ModelType.MINDIR)
            >>> inputs = model.get_inputs()
            >>> for input in inputs:
            ...     in_data = np.arange(1 * 3 * 224 * 224, dtype=np.float32).reshape((1, 3, 224, 224))
            ...     input.set_data_from_numpy(in_data)
            ...
            >>> outputs = model.predict(inputs)
            >>> for output in outputs:
            ...     data = output.get_data_to_numpy()
            ...     print("outputs' shape: ", data.shape)
            ...
            outputs' shape:  (1,1001)
            >>> # 3. predict which indata is from new MindSpore Lite's Tensor with numpy array
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> model = mslite.Model()
            >>> model.build_from_file("mobilenetv2.mindir", mslite.ModelType.MINDIR)
            >>> inputs = model.get_inputs()
            >>> input_tensors = []
            >>> for input in inputs:
            ...     input_tensor = mslite.Tensor()
            ...     input_tensor.dtype = input.dtype
            ...     input_tensor.shape = input.shape
            ...     input_tensor.format = input.format
            ...     input_tensor.name = input.name
            ...     in_data = np.arange(1 * 3 * 224 * 224, dtype=np.float32).reshape((1, 3, 224, 224))
            ...     input_tensor.set_data_from_numpy(in_data)
            ...     input_tensors.append(input_tensor)
            ...
            >>> outputs = model.predict(input_tensors)
            >>> for output in outputs:
            ...     data = output.get_data_to_numpy()
            ...     print("outputs' shape: ", data.shape)
            ...
            outputs' shape:  (1,1001)
        """
        # pylint: disable=useless-super-delegation
        if not isinstance(inputs, (list, tuple)):
            raise TypeError(
                "inputs must be list or tuple, but got {}.".format(type(inputs)))
        model_input_tensors = self.get_inputs()
        if len(model_input_tensors) != len(inputs):
            raise RuntimeError(f"inputs size is wrong.")
        inputs_tensor = []
        for i, in_tensor in enumerate(inputs):
            if isinstance(in_tensor, np.ndarray):
                model_input_tensors[i].set_data_from_numpy(in_tensor)
                inputs_tensor.append(model_input_tensors[i])
            elif isinstance(in_tensor, Tensor):
                inputs_tensor.append(in_tensor)
            else:
                raise TypeError("inputs element must be Tensor, of numpy.")
        return super(Model, self).predict(inputs_tensor, outputs)

    def resize(self, inputs, dims):
        """
        Resizes the shapes of inputs. This method is used in the following scenarios:

        1. If multiple inputs of the same size need to predicted, you can set the batch dimension of `dims` to
           the number of inputs, then multiple inputs can be performed inference at the same time.

        2. Adjust the input size to the specify shape.

        3. When the input is a dynamic shape (a dimension of the shape of the model input contains -1), -1 must be
           replaced by a fixed dimension through `resize` .

        4. The shape operator contained in the model is dynamic shape (a dimension of the shape operator contains -1).

        Args:
            inputs (list[Tensor]): A list that includes all input Tensors in order.
            dims (list[list[int]]): A list that includes the new shapes of input Tensors, should be consistent with
                input Tensors' shape.

        Raises:
            TypeError: `inputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            TypeError: `dims` is not a list.
            TypeError: `dims` is a list, but the elements are not list.
            TypeError: `dims` is a list, the elements are list, but the element's elements are not int.
            ValueError: The size of `inputs` is not equal to the size of `dims` .
            RuntimeError: resize inputs failed.

        Examples:
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> model.build_from_file("mobilenetv2.mindir", mslite.ModelType.MINDIR)
            >>> inputs = model.get_inputs()
            >>> print("Before resize, the first input shape: ", inputs[0].shape)
            Before resize, the first input shape: [1, 3, 224, 224]
            >>> model.resize(inputs, [[1, 3, 112, 112]])
            >>> print("After resize, the first input shape: ", inputs[0].shape)
            After resize, the first input shape: [1, 3, 112, 112]
        """
        # pylint: disable=useless-super-delegation
        super(Model, self).resize(inputs, dims)


class ModelParallelRunner:
    """
    The `ModelParallelRunner` class defines a MindSpore Lite's Runner, which support model parallelism. Compared with
    `model` , `model` does not support parallelism, but `ModelParallelRunner` supports parallelism. A Runner contains
    multiple workers, which are the units that actually perform parallel inferring. The primary use case is when
    multiple clients send inference tasks to the server, the server perform parallel inference, shorten the inference
    time, and then return the inference results to the clients.

    Examples:
        >>> # Use case: serving inference.
        >>> # precondition 1: Building MindSpore Lite serving package by export MSLITE_ENABLE_SERVER_INFERENCE=on.
        >>> # precondition 2: install wheel package of MindSpore Lite built by precondition 1.
        >>> import mindspore_lite as mslite
        >>> model_parallel_runner = mslite.ModelParallelRunner()
        >>> print(model_parallel_runner)
        model_path: .
    """

    def __init__(self):
        if hasattr(_c_lite_wrapper, "ModelParallelRunnerBind"):
            self._model = _c_lite_wrapper.ModelParallelRunnerBind()
        else:
            raise RuntimeError(f"ModelParallelRunner init failed, If you want to use it, you need to build"
                               f"MindSpore Lite serving package by export MSLITE_ENABLE_SERVER_INFERENCE=on.")
        self.model_path_ = ""

    def __str__(self):
        return f"model_path: {self.model_path_}."

    def build_from_file(self, model_path, context=None):
        """
        build a model parallel runner from model path so that it can run on a device.

        Args:
            model_path (str): Define the model path.
            context (Context, optional): Define the config used to transfer context and options during building model.
                Default: ``None``. ``None`` means the Context with cpu target. Context has the default parallel
                attribute.

        Raises:
            TypeError: `model_path` is not a str.
            TypeError: `context` is neither a Context nor ``None``.
            RuntimeError: `model_path` does not exist.
            RuntimeError: ModelParallelRunner's init failed.

        Examples:
            >>> # Use case: serving inference.
            >>> # precondition 1: Building MindSpore Lite serving package by export MSLITE_ENABLE_SERVER_INFERENCE=on.
            >>> # precondition 2: install wheel package of MindSpore Lite built by precondition 1.
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> context.target = ["cpu"]
            >>> context.parallel.workers_num = 4
            >>> model_parallel_runner = mslite.ModelParallelRunner()
            >>> model_parallel_runner.build_from_file(model_path="mobilenetv2.mindir", context=context)
            >>> print(model_parallel_runner)
            model_path: mobilenetv2.mindir.
        """
        check_isinstance("model_path", model_path, str)
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"ModelParallelRunner's build from file failed, model_path does not exist!")
        self.model_path_ = model_path
        if context is None:
            ret = self._model.init(self.model_path_, None)
        else:
            check_isinstance("context", context, Context)
            ret = self._model.init(
                self.model_path_, context.parallel._runner_config)
        if not ret.IsOk():
            raise RuntimeError(
                f"ModelParallelRunner's build from file failed! Error is {ret.ToString()}")

    def get_inputs(self):
        """
        Obtains all input Tensors of the model.

        Returns:
            list[Tensor], the input Tensor list of the model.

        Examples:
            >>> # Use case: serving inference.
            >>> # precondition 1: Building MindSpore Lite serving package by export MSLITE_ENABLE_SERVER_INFERENCE=on.
            >>> # precondition 2: install wheel package of MindSpore Lite built by precondition 1.
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> context.target = ["cpu"]
            >>> context.parallel.workers_num = 4
            >>> model_parallel_runner = mslite.ModelParallelRunner()
            >>> model_parallel_runner.build_from_file(model_path="mobilenetv2.mindir", context=context)
            >>> inputs = model_parallel_runner.get_inputs()
        """
        inputs = []
        for _tensor in self._model.get_inputs():
            inputs.append(Tensor(_tensor))
        return inputs

    def predict(self, inputs, outputs=None):
        """
        Inference ModelParallelRunner.

        Args:
            inputs (list[Tensor]): A list that includes all input Tensors in order.
            outputs (list[Tensor], optional): A list that includes all output Tensors in order,
                this tensor include output data buffer.

        Returns:
            list[Tensor], outputs, the model outputs are filled in the container in sequence.

        Raises:
            TypeError: `inputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            RuntimeError: predict model failed.

        Examples:
            >>> # Use case: serving inference.
            >>> # Precondition 1: Download MindSpore Lite serving package or building MindSpore Lite serving package by
            >>> #                 export MSLITE_ENABLE_SERVER_INFERENCE=on.
            >>> # Precondition 2: Install wheel package of MindSpore Lite built by precondition 1.
            >>> # The result can be find in the tutorial of runtime_parallel_python.
            >>> import time
            >>> from threading import Thread
            >>> import numpy as np
            >>> import mindspore_lite as mslite
            >>> # the number of threads of one worker.
            >>> # WORKERS_NUM * THREAD_NUM should not exceed the number of cores of the machine.
            >>> THREAD_NUM = 1
            >>> # In parallel inference, the number of workers in one `ModelParallelRunner` in server.
            >>> # If you prepare to compare the time difference between parallel inference and serial inference,
            >>> # you can set WORKERS_NUM = 1 as serial inference.
            >>> WORKERS_NUM = 3
            >>> # Simulate 5 clients, and each client sends 2 inference tasks to the server at the same time.
            >>> PARALLEL_NUM = 5
            >>> TASK_NUM = 2
            >>>
            >>>
            >>> def parallel_runner_predict(parallel_runner, parallel_id):
            ...     # One Runner with 3 workers, set model input, execute inference and get output.
            ...     task_index = 0
            ...     while True:
            ...         if task_index == TASK_NUM:
            ...             break
            ...         task_index += 1
            ...         # Set model input
            ...         inputs = parallel_runner.get_inputs()
            ...         in_data = np.fromfile("input.bin", dtype=np.float32)
            ...         inputs[0].set_data_from_numpy(in_data)
            ...         once_start_time = time.time()
            ...         # Execute inference
            ...         outputs = parallel_runner.predict(inputs)
            ...         once_end_time = time.time()
            ...         print("parallel id: ", parallel_id, " | task index: ", task_index, " | run once time: ",
            ...               once_end_time - once_start_time, " s")
            ...         # Get output
            ...         for output in outputs:
            ...             tensor_name = output.name.rstrip()
            ...             data_size = output.data_size
            ...             element_num = output.element_num
            ...             print("tensor name is:%s tensor size is:%s tensor elements num is:%s" % (tensor_name,
            ...                                                                                      data_size,
            ...                                                                                      element_num))
            ...
            ...             data = output.get_data_to_numpy()
            ...             data = data.flatten()
            ...             print("output data is:", end=" ")
            ...             for j in range(5):
            ...                 print(data[j], end=" ")
            ...             print("")
            ...
            >>> # Init RunnerConfig and context, and add CPU device info
            >>> context = mslite.Context()
            >>> context.target = ["cpu"]
            >>> context.cpu.enable_fp16 = False
            >>> context.cpu.thread_num = THREAD_NUM
            >>> context.cpu.inter_op_parallel_num = THREAD_NUM
            >>> context.parallel.workers_num = WORKERS_NUM
            >>> # Build ModelParallelRunner from file
            >>> model_parallel_runner = mslite.ModelParallelRunner()
            >>> model_parallel_runner.build_from_file(model_path="mobilenetv2.mindir", context=context)
            >>> # The server creates 5 threads to store the inference tasks of 5 clients.
            >>> threads = []
            >>> total_start_time = time.time()
            >>> for i in range(PARALLEL_NUM):
            ...     threads.append(Thread(target=parallel_runner_predict, args=(model_parallel_runner, i,)))
            ...
            >>> # Start threads to perform parallel inference.
            >>> for th in threads:
            ...     th.start()
            ...
            >>> for th in threads:
            ...     th.join()
            ...
            >>> total_end_time = time.time()
            >>> print("total run time: ", total_end_time - total_start_time, " s")
        """
        if not isinstance(inputs, (list, tuple)):
            raise TypeError(
                "inputs must be list or tuple, but got {}.".format(type(inputs)))
        _inputs = []
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
            _inputs.append(element._tensor)
        _outputs = []
        if outputs is not None:
            if not isinstance(outputs, list):
                raise TypeError("outputs must be list, bug got {}.".format(type(inputs)))
            for i, element in enumerate(outputs):
                if not isinstance(element, Tensor):
                    raise TypeError(f"outputs element must be Tensor, bug got {type(element)} at index {i}.")
                # pylint: disable=protected-access
                _outputs.append(element._tensor)

        _outputs = self._model.predict(_inputs, _outputs, None, None)
        if not isinstance(_outputs, list) or len(_outputs) == 0:
            raise RuntimeError(f"predict failed!")
        predict_outputs = []
        for _output in _outputs:
            predict_outputs.append(Tensor(_output))
        return predict_outputs


class ModelGroupFlag(Enum):
    """
    The `ModelGroupFlag` class defines the type of the model group.

    The `ModelGroupFlag` is used to define the flags used to construct a `ModelGroup`. Currently, supports:

    1. `ModelGroupFlag.SHARE_WEIGHT`, multiple models share weights share workspace memory, default construction flag
    for `ModelGroup`.

    2. `ModelGroupFlag.SHARE_WORKSPACE`, multiple models share weights(including constatns and variables) memory.
    Currently only supported in cloud side Ascend inference and the provider is GE.

    Examples:
        >>> import mindspore_lite as mslite
        >>> context = mslite.Context()
        >>> context.target = ["Ascend"]
        >>> context.ascend.device_id = 0
        >>> context.ascend.rank_id = 0
        >>> context.ascend.provider = "ge"
        >>> model_group = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT)
        >>> model0 = mslite.Model()
        >>> model1 = mslite.Model()
        >>> model_group.add_model([model0, model1])
        >>> model0.build_from_file("seq_1024.mindir", mslite.ModelType.MINDIR, context, "config0.ini")
        >>> model1.build_from_file("seq_1.mindir", mslite.ModelType.MINDIR, context, "config.ini")
    """

    SHARE_WEIGHT = 1
    SHARE_WORKSPACE = 2


model_group_flag_py_cxx_map = {
    ModelGroupFlag.SHARE_WEIGHT: _c_lite_wrapper.ModelGroupFlag.kShareWeight,
    ModelGroupFlag.SHARE_WORKSPACE: _c_lite_wrapper.ModelGroupFlag.kShareWorkspace,
}


class ModelGroup:
    """
    The `ModelGroup` class is used to define a MindSpore model group,
    facilitating multiple models to share workspace memory or weights(including constants and variables) memory.

    Args:
       flags (ModelGroupFlag, optional): Indicates the type of the model group.
           Default: ``ModelGroupFlag.SHARE_WEIGHT``.

    Examples:
        >>> # Multi models share workspace memory
        >>> import mindspore_lite as mslite
        >>> model_group = mslite.ModelGroup()
        >>> model_group.add_model([path1, path2])
        >>> model_group.cal_max_size_of_workspace(model_type, context)
        >>>
        >>> # Multi models share weights memory
        >>> import mindspore_lite as mslite
        >>> context = mslite.Context()
        >>> context.target = ["Ascend"]
        >>> context.ascend.device_id = 0
        >>> context.ascend.rank_id = 0
        >>> context.ascend.provider = "ge"
        >>> model_group = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT)
        >>> model0 = mslite.Model()
        >>> model1 = mslite.Model()
        >>> model_group.add_model([model0, model1])
        >>> model0.build_from_file("seq_1024.mindir", mslite.ModelType.MINDIR, context, "config0.ini")
        >>> model1.build_from_file("seq_1.mindir", mslite.ModelType.MINDIR, context, "config.ini")
    """

    def __init__(self, flags=ModelGroupFlag.SHARE_WORKSPACE):
        if flags == ModelGroupFlag.SHARE_WORKSPACE:
            flags_inner = _c_lite_wrapper.ModelGroupFlag.kShareWorkspace
        elif flags == ModelGroupFlag.SHARE_WEIGHT:
            flags_inner = _c_lite_wrapper.ModelGroupFlag.kShareWeight
        else:
            raise RuntimeError(
                f"Parameter flags should be ModelGroupFlag.SHARE_WORKSPACE or ModelGroupFlag.SHARE_WEIGHT")
        self._model_group = _c_lite_wrapper.ModelGroupBind(flags_inner)

    def add_model(self, models):
        """
        Used to define MindSpore Lite model grouping information, which is used to share workspace memory or
        weight (including constants and variables) memory. This interface only supports weight memory
        sharing when the `models` is a tuple or list of `Model` objects, and only supports workspace memory sharing in
        other scenarios.

        Args:
           models (union[tuple/list(str), tuple/list(Model)]): Define the list/tuple of model paths or Model objects.

        Raises:
           TypeError: `models` is not a list and tuple.
           TypeError: `models` is a list or tuple, but the elements are not all str or Model.
           RuntimeError: Failed to add model grouping information.
        """
        if not isinstance(models, (list, tuple)):
            raise TypeError(f"models must be list/tuple, but got {type(models)}")
        if not models:
            raise RuntimeError(f"models cannot be empty")
        model0 = models[0]
        if isinstance(model0, str):
            for i, element in enumerate(models):
                if not isinstance(element, str):
                    raise TypeError(f"models element must be all str or Model, but got "
                                    f"{type(element)} at index {i}.")
            ret = self._model_group.add_model(models)
        elif isinstance(model0, Model):
            for i, element in enumerate(models):
                if not isinstance(element, Model):
                    raise TypeError(f"models element must be all str or Model, but got "
                                    f"{type(element)} at index {i}.")
            models_inner = [model._model for model in models]
            ret = self._model_group.add_model_by_object(models_inner)
        else:
            raise TypeError(f"models element must be all str or Model, but got "
                            f"{type(model0)} at index {0}.")
        if not ret.IsOk():
            raise RuntimeError(f"ModelGroup's add model failed.")

    def cal_max_size_of_workspace(self, model_type, context):
        """
        Calculate the max workspace of the added models. Only valid when the type of `ModelGroup` is
        ``ModelGroupFlag.SHARE_WORKSPACE``.

        Args:
           model_type (ModelType): model_type Define The type of model file.
           context (Context): context A context used to store options.

        Raises:
           TypeError: `model_type` is not a ModelType.
           TypeError: `context` is a Context.
           RuntimeError: cal max size of workspace failed.
        """
        check_isinstance("context", context, Context)
        check_isinstance("model_type", model_type, ModelType)
        model_type_ = _c_lite_wrapper.ModelType.kMindIR_Lite
        if model_type is ModelType.MINDIR:
            model_type_ = _c_lite_wrapper.ModelType.kMindIR
        ret = self._model_group.cal_max_size_of_workspace(
            model_type_, context._context._inner_context)
        if not ret.IsOk():
            raise RuntimeError(
                f"ModelGroup's cal max size of workspace failed.")
