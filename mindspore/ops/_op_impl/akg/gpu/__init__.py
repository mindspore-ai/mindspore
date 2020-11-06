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
from .cast import _cast_akg
from .equal import _equal_akg
from .greater_equal import _greater_equal_akg
from .hsigmoid import _hsigmoid_akg
from .hsigmoid_grad import _hsigmoid_grad_akg
from .hswish import _hswish_akg
from .hswish_grad import _hswish_grad_akg
from .inplace_assign import _inplace_assign_akg
from .lessequal import _lessequal_akg
from .logical_and import _logical_and_akg
from .logical_not import _logical_not_akg
from .logical_or import _logical_or_akg
from .mean import _simple_mean_akg
from .mean_grad import _simple_mean_grad_akg
from .mul import _mul_akg
from .notequal import _notequal_akg
from .relu6 import _relu6_akg
from .relu6_grad import _relu6_grad_akg
from .squeeze import _squeeze_akg
from .squeeze_grad import _squeeze_grad_akg
from .sub import _sub_akg
from .tile import _tile_akg
# Please insert op register in lexicographical order of the filename.
