# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Some utils."""
from __future__ import absolute_import
import inspect
from functools import wraps


def cell_attr_register(fn=None, attrs=None):
    """
    Cell init attributes register.

    Registering the decorator of the built-in operator cell __init__
    function will add save all the parameters of __init__ as operator attributes.

    Args:
        fn (function): __init__ function of cell.
        attrs (list(string) | string): attr list.

    Returns:
        function, original function.
    """

    def wrap_cell(fn):
        @wraps(fn)
        def deco(self, *args, **kwargs):
            arguments = []
            if attrs is None:
                bound_args = inspect.signature(fn).bind(self, *args, **kwargs)
                arguments = bound_args.arguments
                del arguments['self']
                arguments = arguments.values()
            fn(self, *args, **kwargs)
            if attrs is None:
                self.cell_init_args = type(self).__name__ + str(arguments)
                return

            if isinstance(attrs, list):
                for item in attrs:
                    if not isinstance(item, str):
                        raise ValueError(f"attr must be a string")
                    if hasattr(self, item):
                        arguments.append(getattr(self, item))
            elif isinstance(attrs, str):
                if hasattr(self, attrs):
                    arguments = getattr(self, attrs)
            else:
                raise ValueError(f"attrs must be list or string")
            self.cell_init_args = type(self).__name__ + str(arguments)

        return deco

    if fn is not None:
        return wrap_cell(fn)
    return wrap_cell
