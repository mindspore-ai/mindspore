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
from mindspore.ops import prim_attr_register, PrimitiveWithInfer


# sum = input1 + input2 + const_bias
class CusAdd3(PrimitiveWithInfer):
    """Custom add3 definition"""

    @prim_attr_register
    def __init__(self, const_bias=0.0):
        self.init_prim_io_names(inputs=['input1', 'input2'], outputs=['sum3'])
        from add3_impl import CusAdd3Impl

    def infer_shape(self, input1, input2):
        return input1

    def infer_dtype(self, input1, input2):
        return input1
