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
            Supports ["O0", "O1", "O2"]. Default: "O1".

            - "O0": Basic optimization.
            - "O1": Manual optimization.
            - "O2": Manual optimization and graph computation fusion.

        task_sink (bool): Determines whether to pass the data through dataset channel. Default: True.
        **kwargs (dict): A dictionary of keyword arguments that the class needs.
    """
    def __init__(self, jit_level="O1", task_sink=True, **kwargs):
        if jit_level not in ["O0", "O1", "O2"]:
            raise ValueError("For 'jit_level' must be one of ['O0', 'O1', 'O2'].")
        if not isinstance(task_sink, bool):
            raise TypeError("For 'task_sink' must be bool.")
        self.jit_config_dict = dict()
        self.jit_config_dict["jit_level"] = jit_level
        self.jit_config_dict["task_sink"] = str(int(task_sink))
        for key, value in kwargs.items():
            self.jit_config_dict[key] = value
