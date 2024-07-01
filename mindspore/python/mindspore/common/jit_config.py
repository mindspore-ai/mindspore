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

    Args:
        jit_level (str, optional): Used to control the compilation optimization level.
            Supports ["O0", "O1", "O2"]. Default: ``""`` , The framework automatically selects the execution method.
            Not recommended, it is recommended to use the jit decorator.

            - ``"O0"``: Except for optimizations that may affect functionality, all other optimizations are turned off,
              adopt KernelByKernel execution mode.
            - ``"O1"``: Using commonly used optimizations and automatic operator fusion optimizations,
              adopt KernelByKernel execution mode.
            - ``"O2"``: Ultimate performance optimization, adopt Sink execution mode.

        exc_mode (str, optional): Control the execution mode of the model.
            Supports ["auto", "sink", "no_sink"]. Default: ``"auto"`` .

            - ``"auto"``: The framework automatically selects the execution method.
            - ``"sink"``: Support the network to load and load the entire device at once, and then execute it by
              input driver, without the need to iterate through each operator to achieve better execution performance.
              This mode is only supported on the Ascend backend.
            - ``"no_sink"``: The network model is executed asynchronously one by one using a single operator.

        jit_syntax_level (str, optional): JIT syntax level for graph compiling.
            The value must be ``"STRICT"`` , ``"LAX"`` or ``""`` . Default to an empty string, which means that this
            JitConfig configuration will be ignored and the jit_syntax_level of ms.context will be used.
            For more details about ms.context, refer to
            `set_context <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_context.html>`_ .
            Default: ``""`` .

            - ``"STRICT"``: Only basic syntax is supported, and execution performance is optimal. Can be used for MindIR
              load and export.
            - ``"LAX"``: Compatible with all Python syntax as much as possible. However, execution performance may be
              affected and not optimal. Cannot be used for MindIR load and export due to some syntax that may not be
              able to be exported.

        debug_level (str, optional): Set debugging level for graph compiling.
            The value must be ``"RELEASE"`` or ``"DEBUG"``. Default value: ``RELEASE``.

            - ``RELEASE``: Used for normally running, and some debug information will be discard to get a better
              compiling performance.
            - ``DEBUG``: Used for debugging when errors occur, more information will be record in compiling process.

        infer_boost (str, optional): enable infer boost mode.
            The value must be ``"on"`` , ``"off"``. Default to an "off", which means that disable infer boost.
            when infer boost mode is enabled, MindSpore will use high perf kernel lib, use faster runtime make
            infer speed is best.
            Note: current infer boost only support `jit_level` == ``"O0"`` and only Atlas A2 series products
            are supported.

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
    def __init__(self, jit_level="", exc_mode="auto", jit_syntax_level="", debug_level="RELEASE",
                 infer_boost="off", **kwargs):
        if jit_level not in ["", "O0", "O1", "O2"]:
            raise ValueError("For 'jit_level' must be one of ['O0', 'O1', 'O2'].")
        if exc_mode not in ["auto", "sink", "no_sink"]:
            raise ValueError("For 'exc_mode' must be one of '['auto', 'sink', 'no_sink']'.")
        if jit_syntax_level != "" and jit_syntax_level not in ["STRICT", "COMPATIBLE", "LAX"]:
            raise ValueError("For 'jit_syntax_level' must be one of '['STRICT', 'LAX']'.")
        if debug_level not in ["RELEASE", "DEBUG"]:
            raise ValueError("For 'debug_level' must be one of '['RELEASE', 'DEBUG']'.")
        if infer_boost != "" and infer_boost not in ["on", "off"]:
            raise ValueError("For 'infer_boost' must be one of '['on', 'off']'.")
        self.jit_config_dict = kwargs
        self.jit_config_dict["jit_level"] = jit_level
        self.jit_config_dict["exc_mode"] = exc_mode
        self.jit_config_dict["jit_syntax_level"] = jit_syntax_level
        self.jit_config_dict["debug_level"] = debug_level
        self.jit_config_dict["infer_boost"] = infer_boost
