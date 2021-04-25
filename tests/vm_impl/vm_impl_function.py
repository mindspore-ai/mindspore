# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Generate vm_impl function for nn ops without python object"""
from mindspore.common.tensor import Tensor
from .vm_interface import vm

def ReluGrad(y_backprop, x):
    x = x.asnumpy()
    y_backprop = y_backprop.asnumpy()
    y_backprop = vm.relu_grad(x.copy()) * y_backprop
    return Tensor(y_backprop)
