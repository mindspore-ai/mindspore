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
"""JitConfig for compile."""


class JitConfig:
    """
    Jit config for compile.

    Note:
        This is an experimental function that is subject to change or deletion.

    Args:
        jit_level (str): Option for argument `level` for Optimization of lift graph.
            Supports ["O0", "O1", "O2", "O3"]. Default: "O1".

            - "O0": Basic optimization.
            - "O1": Manual optimization.
            - "O2": Manual optimization and graph computation fusion.
            - "O3": Performance optimization, no generalization guaranteed.

        exc_mode (str): Mode for execute the network. Supports ["auto", "sink", "no_sink"]. Default: "auto".

            - "auto": Automatic Policies.
            - "sink": Build computational graphs with the sink mode.
            - "no_sink": Build computational graphs with no sink mode.

        **kwargs (dict): A dictionary of keyword arguments that the class needs.

    Examples:
        >>> from mindspore import JitConfig
        >>>
        >>> jitconfig = JitConfig(jit_level="O1")
        >>> net = LeNet5()
        >>>
        >>> net.set_jit_config(jitconfig)
    """
    def __init__(self, jit_level="O1", exc_mode="auto", **kwargs):
        if jit_level not in ["O0", "O1", "O2", "O3"]:
            raise ValueError("For 'jit_level' must be one of ['O0', 'O1', 'O2', 'O3'].")
        if exc_mode not in ['auto', 'sink', 'no_sink']:
            raise ValueError("For 'exc_mode' must be one of '['auto', 'sink', 'no_sink']'.")
        self.jit_config_dict = kwargs
        self.jit_config_dict["jit_level"] = jit_level
        self.jit_config_dict["exc_mode"] = exc_mode
