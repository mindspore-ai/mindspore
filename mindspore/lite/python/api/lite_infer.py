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
LiteInfer API.
"""
from __future__ import absolute_import
import importlib

from mindspore_lite.context import Context
from mindspore_lite.lib import _c_lite_wrapper
from mindspore_lite._checkparam import check_isinstance
from .base_model import BaseModel

__all__ = ['LiteInfer']


class LiteInfer(BaseModel):
    """
    The LiteInfer class takes training model as input and performs predictions directly.

    Args:
        model_or_net (Model, Cell): MindSpore Model or MindSpore nn.Cell.
        net_inputs (Union[Tensor, Dataset, List, Tuple, Number, Bool]): It represents the inputs
             of the `net`, if the network has multiple inputs, set them together. While its type is Dataset,
             it represents the preprocess behavior of the `net`, data preprocess operations will be serialized.
             In second situation, you should adjust batch size of dataset script manually which will impact on
             the batch size of 'net' input. Only supports parse "image" column from dataset currently.
        context (Context, optional): Define the context used to transfer options during execution. Default: ``None``.
                None means the Context with cpu target.
        model_group_id (int, optional) :  model_group_id is used to bind model to model group
        config (dict, optional): Enabled when the backend is 'lite'. config includes two parts,
                config_path ('configPath', str) and config_item (str, dict). When config_item is set,
                its priority is higher than config_path. Set rank table file for inference. The content
                of the configuration file is as follows:
                .. code-block::
                    [ascend_context]
                    rank_table_file=[path_a](storage initial path of the rank table file)
                When set
                .. code-block::
                    config = {"ascend_context" : {"rank_table_file" : "path_b"}}
                The path_b from the config will be used to compile the model.

    Raises:
        ValueError: `train_model` is not a MindSpore Model.
    """

    def __init__(self, model_or_net, *net_inputs, context=None, model_group_id=None, config: dict = None):
        super(LiteInfer, self).__init__(_c_lite_wrapper.LiteInferPyBind())
        self._mindspore = None
        # pylint: disable=broad-except
        try:
            self._mindspore = importlib.import_module('mindspore')
        except (ImportError, BaseException):
            raise ImportError("For 'LiteInfer', import mindspore fail.")

        self._infer_network = None
        if isinstance(model_or_net, self._mindspore.train.Model):
            self._infer_network = model_or_net.predict_network
        elif isinstance(model_or_net, self._mindspore.nn.Cell):
            self._infer_network = model_or_net
        else:
            raise ValueError(f"For LiteInfer, input model_or_net should be ms.train.Model or "
                             f"ms.nn.Cell, but got {type(model_or_net)}.")

        self._load_and_update_config(config)

        if model_group_id is not None:
            # bind model to model group by model_group_id
            self._model.bind_model_to_modelgroup(model_group_id)

        import os
        # ms train need enable refmode
        self._func_graph = self._get_func_graph(self._infer_network, *net_inputs)
        # ms lite need disable refmode
        os.environ["MS_DISABLE_REF_MODE"] = "1"
        self._build_from_fun_graph(self._func_graph, context)

    def _load_and_update_config(self, config: dict = None):
        """
        Load and update config.
        """
        config_path = "configPath"
        if config:
            check_isinstance("config", config, dict)
            for k, v in config.items():
                check_isinstance("config_key", k, str)
                if k == config_path:
                    check_isinstance("config_value", v, str)
                    self._model.load_config(v)

            for k, v in config.items():
                check_isinstance("config_key", k, str)
                if k != config_path:
                    check_isinstance("config_value", v, dict)
                    for v_k, v_v in v.items():
                        check_isinstance("config_value_key", v_k, str)
                        check_isinstance("config_value_value", v_v, str)
                        ret = self._model.update_config(k, v)
                        if not ret.IsOk():
                            raise RuntimeError(f"update configuration failed! Error is {ret.ToString()}."
                                               f"Setcion is {k}, config is {v}")

    def _get_func_graph(self, pyobj, *net_inputs):
        """
        Get Func graph from frontend compile.

        Return: a _c_expression FunGraph object.
        """
        # pylint: disable=protected-access
        return self._mindspore.train.serialization._get_funcgraph(pyobj, *net_inputs)

    def get_inputs(self):
        """
        Obtains all input Tensors of the model.

        See `mindspore_lite.model.get_inputs` for more details.
        """
        # pylint: disable=useless-super-delegation
        return super(LiteInfer, self).get_inputs()

    def predict(self, inputs):
        """
        Inference model.

        See `mindspore_lite.model.predict` for more details.
        """
        # pylint: disable=useless-super-delegation
        return super(LiteInfer, self).predict(inputs)

    def resize(self, inputs, dims):
        """
        Resizes the shapes of inputs.

        See `mindspore_lite.model.resize` for more details.
        """
        # pylint: disable=useless-super-delegation
        super(LiteInfer, self).resize(inputs, dims)

    def _build_from_fun_graph(self, func_graph, context):
        """
        Build from funcgraph
        """
        if context is None:
            context = Context()
        check_isinstance("context", context, Context)
        # pylint: disable=protected-access
        ret = self._model.build_from_func_graph(func_graph, context._context._inner_context)
        if not ret.IsOk():
            raise RuntimeError(f"build_from_file failed! Error is {ret.ToString()}")
