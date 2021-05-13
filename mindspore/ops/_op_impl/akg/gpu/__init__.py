# Copyright 2019 Huawei Technologies Co., Ltd
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

"""__init__"""
from .equal import _equal_akg
from .greater_equal import _greater_equal_akg
from .inplace_assign import _inplace_assign_akg
from .lessequal import _lessequal_akg
from .logical_and import _logical_and_akg
from .logical_not import _logical_not_akg
from .logical_or import _logical_or_akg
from .mean import _simple_mean_akg
from .mean_grad import _simple_mean_grad_akg
from .notequal import _notequal_akg
# Please insert op register in lexicographical order of the filename.
