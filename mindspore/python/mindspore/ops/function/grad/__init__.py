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

"""Defines gradient related operators with functional form."""

from __future__ import absolute_import
from .grad_func import (
    grad,
    value_and_grad,
    jacfwd,
    jacrev,
    jet,
    derivative,
    jvp,
    vjp,
    linearize,
    stop_gradient
)

__all__ = []
__all__.extend(grad_func.__all__)
