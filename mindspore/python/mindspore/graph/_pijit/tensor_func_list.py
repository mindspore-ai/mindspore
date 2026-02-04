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
"""Store and get tensor method"""
from mindspore import Tensor
from mindspore._c_expression import function_id

tensor_method_id_to_name = {}
for method_name in dir(Tensor):
    method_id = function_id(getattr(Tensor, method_name))
    tensor_method_id_to_name[method_id] = method_name


def get_tensor_method_name(id):
    """Get method name by function id"""
    return tensor_method_id_to_name.get(id, None)
