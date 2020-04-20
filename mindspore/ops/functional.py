# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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

"""The names of functional part are summarized here."""

from mindspore.common._register_for_tensor import tensor_operator_registry
from .primitive import Primitive
from . import operations as P
from .operations import _grad_ops

typeof = Primitive('typeof')
hastype = Primitive('hastype')
cast = P.Cast()
dtype = P.DType()


issubclass_ = P.IsSubClass()
isinstance_ = P.IsInstance()
fill = P.Fill()
select = P.Select()
size = P.Size()
ones_like = P.OnesLike()
shape = P.Shape()
rank = P.Rank()
reshape = P.Reshape()
# control_depend: represent dependency between two operators
control_depend = P.ControlDepend()
merge = P.Merge()
geswitch = P.GeSwitch()
addn = P.AddN()
tensor_add = P.TensorAdd()
neg_tensor = P.Neg()
tensor_lt = P.Less()
tensor_le = P.LessEqual()
tensor_gt = P.Greater()
tensor_ge = P.GreaterEqual()
tensor_sub = P.Sub()
tensor_mul = P.Mul()
tensor_div = P.RealDiv()
tensor_floordiv = P.FloorDiv()
tensor_pow = P.Pow()
tensor_mod = P.FloorMod()
strided_slice = P.StridedSlice()
same_type_shape = P.SameTypeShape()
equal = P.Equal()
not_equal = P.NotEqual()
assign_sub = P.AssignSub()
assign = P.Assign()
square = P.Square()
sqrt = P.Sqrt()
scalar_to_array = P.ScalarToArray()
scalar_to_tensor = P.ScalarToTensor()
tuple_to_array = P.TupleToArray()
scalar_cast = P.ScalarCast()


tuple_setitem = Primitive('tuple_setitem')
tuple_getitem = Primitive('tuple_getitem')
list_getitem = Primitive('list_getitem')
list_setitem = Primitive('list_setitem')
dict_getitem = Primitive('dict_getitem')
dict_setitem = Primitive('dict_setitem')
tuple_div = Primitive("tuple_div")
tuple_len = Primitive("tuple_len")
tuple_reversed = Primitive("tuple_reversed")
make_range = Primitive("make_range")
make_tuple = Primitive('make_tuple')
make_dict = Primitive('make_dict')
make_list = Primitive('make_list')
make_slice = Primitive('make_slice')
tuple_equal = Primitive("tuple_equal")
list_equal = Primitive("list_equal")
make_ref = Primitive("make_ref")


scalar_add = Primitive('scalar_add')
scalar_mul = Primitive('scalar_mul')
scalar_sub = Primitive('scalar_sub')
scalar_div = Primitive('scalar_div')
scalar_floordiv = Primitive('scalar_floordiv')
scalar_log = Primitive('scalar_log')
scalar_pow = Primitive('scalar_pow')
scalar_gt = Primitive('scalar_gt')
scalar_ge = Primitive('scalar_ge')
scalar_le = Primitive('scalar_le')
scalar_lt = Primitive('scalar_lt')
scalar_eq = Primitive('scalar_eq')
scalar_ne = Primitive('scalar_ne')
scalar_uadd = Primitive('scalar_uadd')
scalar_usub = Primitive('scalar_usub')
scalar_mod = Primitive('scalar_mod')
string_eq = Primitive('string_equal')
string_concat = Primitive('string_concat')
bool_not = Primitive("bool_not")
bool_or = Primitive("bool_or")
bool_and = Primitive("bool_and")
logical_and = P.LogicalAnd()
logical_or = P.LogicalOr()
logical_not = P.LogicalNot()
array_to_scalar = Primitive('array_to_scalar')
is_ = Primitive("is_")
is_not = Primitive("is_not")
in_dict = Primitive("in_dict")
not_in_dict = Primitive("not_in_dict")
broadcast_gradient_args = Primitive('BroadcastGradientArgs')
dot = Primitive('dot')
array_reduce = Primitive('array_reduce')
partial = Primitive('partial')
zeros_like_tensor = Primitive('zeros_like_tensor')
identity = Primitive('identity')
distribute = Primitive('distribute')
# depend: mount a node to another node
depend = Primitive('depend')
embed = Primitive('embed')
ref_to_embed = _grad_ops.RefToEmbed()
env_setitem = Primitive('env_setitem')
env_getitem = Primitive('env_getitem')
env_add = Primitive('env_add')
J = Primitive('J')
switch = Primitive('switch')
# for sum bprop
reduced_shape = Primitive("reduced_shape")
# shape_mul:input mush be shape multiply elemts in tuple(shape)
shape_mul = Primitive("shape_mul")
# a primitive to compare between tuple.
stop_gradient = Primitive("stop_gradient")

tensor_operator_registry.register('__add__', tensor_add)
tensor_operator_registry.register('__mul__', tensor_mul)
tensor_operator_registry.register('__div__', tensor_div)
