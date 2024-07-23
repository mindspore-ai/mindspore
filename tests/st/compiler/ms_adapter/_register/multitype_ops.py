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

from mindspore.ops.composite.multitype_ops.add_impl import add
from mindspore.ops.composite.multitype_ops.sub_impl import sub
from mindspore.ops.composite.multitype_ops.mul_impl import mul
from mindspore.ops.composite.multitype_ops.div_impl import div
from mindspore.ops.composite.multitype_ops.floordiv_impl import floordiv
from mindspore.ops.composite.multitype_ops.mod_impl import mod
from mindspore.ops.composite.multitype_ops.pow_impl import pow_
from mindspore.ops.composite.multitype_ops.bitwise_and_impl import bitwise_and
from mindspore.ops.composite.multitype_ops.bitwise_or_impl import bitwise_or
from mindspore.ops.composite.multitype_ops.bitwise_xor_impl import bitwise_xor
from mindspore.ops.composite.multitype_ops.negative_impl import negative
from mindspore.ops.composite.multitype_ops.logic_not_impl import logical_not
from mindspore.ops.composite.multitype_ops.equal_impl import equal
from mindspore.ops.composite.multitype_ops.not_equal_impl import not_equal
from mindspore.ops.composite.multitype_ops.less_impl import less
from mindspore.ops.composite.multitype_ops.greater_impl import greater
from mindspore.ops.composite.multitype_ops.less_equal_impl import less_equal
from mindspore.ops.composite.multitype_ops.greater_equal_impl import greater_equal
from mindspore.ops.composite.multitype_ops.in_impl import in_
from mindspore.ops.composite.multitype_ops.not_in_impl import not_in_
from mindspore.ops.composite.multitype_ops.getitem_impl import getitem
from mindspore.ops.composite.multitype_ops.setitem_impl import setitem
from tests.st.compiler.ms_adapter._register import utils


# multitype_ops.add
utils.update_multitype_ops_tensor_tensor(add)
utils.update_multitype_ops_number_tensor(add)
utils.update_multitype_ops_tensor_number(add)
utils.update_multitype_ops_tuple_tensor(add)
utils.update_multitype_ops_tensor_tuple(add)
utils.update_multitype_ops_list_tensor(add)
utils.update_multitype_ops_tensor_list(add)

# multitype_ops.sub
utils.update_multitype_ops_tensor_tensor(sub)
utils.update_multitype_ops_number_tensor(sub)
utils.update_multitype_ops_tensor_number(sub)
utils.update_multitype_ops_tuple_tensor(sub)
utils.update_multitype_ops_tensor_tuple(sub)
utils.update_multitype_ops_list_tensor(sub)
utils.update_multitype_ops_tensor_list(sub)

# multitype_ops.mul
utils.update_multitype_ops_tensor_tensor(mul)
utils.update_multitype_ops_number_tensor(mul)
utils.update_multitype_ops_tensor_number(mul)
utils.update_multitype_ops_tuple_tensor(mul)
utils.update_multitype_ops_tensor_tuple(mul)
utils.update_multitype_ops_list_tensor(mul)
utils.update_multitype_ops_tensor_list(mul)

# multitype_ops.div
utils.update_multitype_ops_tensor_tensor(div)
utils.update_multitype_ops_number_tensor(div)
utils.update_multitype_ops_tensor_number(div)
utils.update_multitype_ops_tuple_tensor(div)
utils.update_multitype_ops_tensor_tuple(div)
utils.update_multitype_ops_list_tensor(div)
utils.update_multitype_ops_tensor_list(div)

# multitype_ops.floordiv
utils.update_multitype_ops_tensor_tensor(floordiv)
utils.update_multitype_ops_number_tensor(floordiv)
utils.update_multitype_ops_tensor_number(floordiv)
utils.update_multitype_ops_tuple_tensor(floordiv)
utils.update_multitype_ops_tensor_tuple(floordiv)
utils.update_multitype_ops_list_tensor(floordiv)
utils.update_multitype_ops_tensor_list(floordiv)

# multitype_ops.mod
utils.update_multitype_ops_tensor_tensor(mod)
utils.update_multitype_ops_number_tensor(mod)
utils.update_multitype_ops_tensor_number(mod)
utils.update_multitype_ops_tuple_tensor(mod)
utils.update_multitype_ops_tensor_tuple(mod)
utils.update_multitype_ops_list_tensor(mod)
utils.update_multitype_ops_tensor_list(mod)

# multitype_ops.pow_
utils.update_multitype_ops_tensor_tensor(pow_)
utils.update_multitype_ops_number_tensor(pow_)
utils.update_multitype_ops_tensor_number(pow_)
utils.update_multitype_ops_tuple_tensor(pow_)
utils.update_multitype_ops_tensor_tuple(pow_)
utils.update_multitype_ops_list_tensor(pow_)
utils.update_multitype_ops_tensor_list(pow_)

# multitype_ops.bitwise_and
utils.update_multitype_ops_tensor_tensor(bitwise_and)
utils.update_multitype_ops_number_tensor(bitwise_and)
utils.update_multitype_ops_tensor_number(bitwise_and)

# multitype_ops.bitwise_or
utils.update_multitype_ops_tensor_tensor(bitwise_or)
utils.update_multitype_ops_number_tensor(bitwise_or)
utils.update_multitype_ops_tensor_number(bitwise_or)

# multitype_ops.bitwise_xor
utils.update_multitype_ops_tensor_tensor(bitwise_xor)
utils.update_multitype_ops_number_tensor(bitwise_xor)
utils.update_multitype_ops_tensor_number(bitwise_xor)

# multitype_ops.negative
utils.update_multitype_ops_tensor(negative)

# multitype_ops.logical_not
utils.update_multitype_ops_tensor(logical_not)

# multitype_ops.equal
utils.update_multitype_ops_tensor_tensor(equal)
utils.update_multitype_ops_number_tensor(equal)
utils.update_multitype_ops_tensor_number(equal)

# multitype_ops.not_equal
utils.update_multitype_ops_tensor_tensor(not_equal)
utils.update_multitype_ops_number_tensor(not_equal)
utils.update_multitype_ops_tensor_number(not_equal)

# multitype_ops.less
utils.update_multitype_ops_tensor_tensor(less)
utils.update_multitype_ops_number_tensor(less)
utils.update_multitype_ops_tensor_number(less)

# multitype_ops.greater
utils.update_multitype_ops_tensor_tensor(greater)
utils.update_multitype_ops_number_tensor(greater)
utils.update_multitype_ops_tensor_number(greater)

# multitype_ops.less_equal
utils.update_multitype_ops_tensor_tensor(less_equal)
utils.update_multitype_ops_number_tensor(less_equal)
utils.update_multitype_ops_tensor_number(less_equal)

# multitype_ops.greater_equal
utils.update_multitype_ops_tensor_tensor(greater_equal)
utils.update_multitype_ops_number_tensor(greater_equal)
utils.update_multitype_ops_tensor_number(greater_equal)

# multitype_ops.in_
utils.update_multitype_ops_tensor_tuple(in_)
utils.update_multitype_ops_tensor_list(in_)

# multitype_ops.not_in_
utils.update_multitype_ops_tensor_tuple(not_in_)
utils.update_multitype_ops_tensor_list(not_in_)

# multitype_ops.getitem
utils.update_multitype_ops_tensor_tensor(getitem)
utils.update_multitype_ops_tensor_number(getitem)
utils.update_multitype_ops_tensor_tuple(getitem)
utils.update_multitype_ops_tensor_list(getitem)
utils.update_multitype_ops_tensor_none(getitem)
utils.update_multitype_ops_tensor_slice(getitem)

# multitype_ops.setitem
utils.update_multitype_ops_setitem_tensor(setitem)
