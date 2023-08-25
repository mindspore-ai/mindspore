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
"""JitConfig for compile."""


class JitConfig:
    """
    Jit config for compile.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        jit_level (str, optional): Option for argument `level` for Optimization of lift graph.
            Supports ["O0", "O1", "O2", "O3"]. Default: ``"O1"`` .

            - "O0": Basic optimization.
            - "O1": Manual optimization.
            - "O2": Manual optimization and graph computation fusion.
            - "O3": Performance optimization, no generalization guaranteed.

        exc_mode (str, optional): Mode for execute the network.
            Supports ["auto", "sink", "no_sink"]. Default: ``"auto"`` .

            - "auto": Automatic Policies.
            - "sink": Build computational graphs with the sink mode.
            - "no_sink": Build computational graphs with no sink mode.

        jit_syntax_level (str, optional): JIT syntax level for graph compiling.
            The value must be ``"STRICT"`` , ``"LAX"`` or ``""`` . Default to an empty string, which means that this
            JitConfig configuration will be ignored and the jit_syntax_level of ms.context will be used.
            For more details about ms.context, refer to
            `set_context <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_context.html>`_ .
            Default: ``""`` .

            - "STRICT": Only basic syntax is supported, and execution performance is optimal. Can be used for MindIR
              load and export.
            - "LAX": Compatible with all Python syntax as much as possible. However, execution performance may be
              affected and not optimal. Cannot be used for MindIR load and export due to some syntax that may not be
              able to be exported.

        **kwargs (dict): A dictionary of keyword arguments that the class needs.

    Examples:
        >>> from mindspore import JitConfig
        >>>
        >>> jitconfig = JitConfig(jit_level="O1")
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>>
        >>> net.set_jit_config(jitconfig)
    """
    def __init__(self, jit_level="O1", exc_mode="auto", jit_syntax_level="", **kwargs):
        if jit_level not in ["O0", "O1", "O2", "O3"]:
            raise ValueError("For 'jit_level' must be one of ['O0', 'O1', 'O2', 'O3'].")
        if exc_mode not in ['auto', 'sink', 'no_sink']:
            raise ValueError("For 'exc_mode' must be one of '['auto', 'sink', 'no_sink']'.")
        if jit_syntax_level != "" and jit_syntax_level not in ['STRICT', 'COMPATIBLE', 'LAX']:
            raise ValueError("For 'jit_syntax_level' must be one of '['STRICT', 'LAX']'.")
        self.jit_config_dict = kwargs
        self.jit_config_dict["jit_level"] = jit_level
        self.jit_config_dict["exc_mode"] = exc_mode
        self.jit_config_dict["jit_syntax_level"] = jit_syntax_level
