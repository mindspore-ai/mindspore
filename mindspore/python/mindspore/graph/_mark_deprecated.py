# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Mark module as deprecated"""

import importlib
import sys
from mindspore import log as logger


def _ensure_module_imported(module_name):
    if module_name not in sys.modules:
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Import '{module_name}' failed: {e}")
    return sys.modules[module_name]


def mark_deprecated(deprecated_module_name, new_module_name, version='2.9.0'):
    deprecated_module = _ensure_module_imported(deprecated_module_name)
    new_module = _ensure_module_imported(new_module_name)
    module_attrs = {k: v for k, v in vars(new_module).items() if not k.startswith('__')}
    deprecated_module.__dict__.update(module_attrs)

    if hasattr(new_module, '__all__'):
        deprecated_module.__all__ = new_module.__all__

    logger.warning(f"Module '{deprecated_module_name}' is deprecated from version {version} and will be removed "
                   f"in a future version, use '{new_module_name}' instead.")
