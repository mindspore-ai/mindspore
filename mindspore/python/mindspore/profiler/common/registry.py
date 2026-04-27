# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""CPU platform profiler."""
from typing import Any, Dict
from mindspore import log as logger


class Register:
    """
    Common register class
    """
    def __init__(self, name) -> None:
        self.module_dict = {}
        self.name = name

    def register_module(self, name):
        """ registers module decorator."""
        def decorator(target):
            model_name = target.__name__
            if name in self.module_dict:
                raise Exception(f'{model_name} already registered with name: {name}')

            self.module_dict[name] = target
            logger.info(f'registered module: {model_name} with name: {name}')
            return target

        return decorator

    @property
    def modules(self) -> Dict[str, Any]:
        """ get all registered modules """
        return self.module_dict


PROFILERS = Register('PROFILERS')
