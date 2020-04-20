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

"""constexpr util"""

from ...primitive import constexpr


@constexpr
def is_same_type(inst, type_):
    """
    Check whether an object is an instance of a target type.

    Inputs:
        inst (mindspore.dtype): Inspected type.
        type_ (mindspore.dtype): Target type.

    Outputs:
        bool, the check result.
    """
    return inst == type_


@constexpr
def error_msg(msg="", format_values=""):
    """
    Used to throw exception information.

    Inputs:
        msg (str): information content.
    """

    raise ValueError(msg.format(*format_values))
