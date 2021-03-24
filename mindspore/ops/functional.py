# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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

"""The names of functional part are summarized here."""

from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.ops import _constants
from .primitive import Primitive
from . import operations as P
from .operations import _grad_ops

typeof = Primitive('typeof')
hastype = Primitive('hastype')
cast = P.Cast()
dtype = P.DType()
isconstant = Primitive('is_constant')
isconstant.set_const_prim(True)

issubclass_ = P.IsSubClass()
isinstance_ = P.IsInstance()
eye = P.Eye()
fill = P.Fill()
tile = P.Tile()
select = P.Select()
size = P.Size()
ones_like = P.OnesLike()
shape = P.Shape()
rank = P.Rank()
reshape = P.Reshape()

merge = P.Merge()
geswitch = P.GeSwitch()
addn = P.AddN()
absolute = P.Abs()
tensor_add = P.Add()
add = tensor_add
neg_tensor = P.Neg()
tensor_lt = P.Less()
less = tensor_lt
tensor_le = P.LessEqual()
le = tensor_le
tensor_gt = P.Greater()
gt = tensor_gt
tensor_ge = P.GreaterEqual()
ge = tensor_ge
tensor_sub = P.Sub()
sub = tensor_sub
tensor_mul = P.Mul()
mul = tensor_mul
tensor_div = P.RealDiv()
div = tensor_div
tensor_floordiv = P.FloorDiv()
floordiv = tensor_floordiv
tensor_pow = P.Pow()
pows = tensor_pow
tensor_mod = P.FloorMod()
floormod = tensor_mod
tensor_exp = P.Exp()
exp = tensor_exp
tensor_expm1 = P.Expm1()
tensor_slice = P.Slice()
strided_slice = P.StridedSlice()
same_type_shape = P.SameTypeShape()
check_bprop = P.CheckBprop()
equal = P.Equal()
not_equal = P.NotEqual()
isfinite = P.IsFinite()
assign_sub = P.AssignSub()
assign_add = P.AssignAdd()
assign = P.Assign()
square = P.Square()
sqrt = P.Sqrt()
log = P.Log()
reduce_sum = P.ReduceSum()
tensor_slice = P.Slice()
maximum = P.Maximum()
minimum = P.Minimum()
floor = P.Floor()
logical_not = P.LogicalNot()
logical_or = P.LogicalOr()
logical_and = P.LogicalAnd()
sin = P.Sin()
cos = P.Cos()
tan = P.Tan()
asin = P.Asin()
acos = P.ACos()
atan = P.Atan()
sinh = P.Sinh()
cosh = P.Cosh()
tanh = P.Tanh()
asinh = P.Asinh()
acosh = P.Acosh()
atanh = P.Atanh()
atan2 = P.Atan2()

scalar_to_array = P.ScalarToArray()
scalar_to_tensor = P.ScalarToTensor()
tuple_to_array = P.TupleToArray()
scalar_cast = P.ScalarCast()
print_ = P.Print()
expand_dims = P.ExpandDims()
transpose = P.Transpose()
squeeze = P.Squeeze()
scatter_nd = P.ScatterNd()
gather = P.Gather()
gather_d = P.GatherD()
gather_nd = P.GatherNd()
scatter_update = P.ScatterUpdate()
scatter_nd_update = P.ScatterNdUpdate()
stack = P.Stack()

def pack(x):
    print("WARNING: 'pack' is deprecated from version 1.1 and will be removed in a future version, use 'stack' instead"
          ".")
    return stack(x)

partial = P.Partial()
# depend: mount a node to another node
depend = P.Depend()
identity = P.identity()

tuple_setitem = Primitive('tuple_setitem')
tuple_getitem = Primitive(_constants.kTupleGetItem)
list_getitem = Primitive('list_getitem')
list_setitem = Primitive('list_setitem')
dict_getitem = Primitive('dict_getitem')
dict_setitem = Primitive('dict_setitem')
tuple_div = Primitive("tuple_div")
tuple_len = Primitive("tuple_len")
list_len = Primitive("list_len")
tuple_reversed = Primitive("tuple_reversed")
make_range = Primitive("make_range")
make_tuple = Primitive('MakeTuple')
make_dict = Primitive('make_dict')
make_list = Primitive('make_list')
make_slice = Primitive('make_slice')
tuple_equal = Primitive("tuple_equal")
list_equal = Primitive("list_equal")
make_ref = Primitive("make_ref")

scalar_add = Primitive(_constants.kScalarAdd)
scalar_mul = Primitive(_constants.kScalarMul)
scalar_sub = Primitive(_constants.kScalarSub)
scalar_div = Primitive(_constants.kScalarDiv)
scalar_floordiv = Primitive(_constants.kScalarFloordiv)
scalar_log = Primitive('scalar_log')
scalar_pow = Primitive(_constants.kScalarPow)
scalar_gt = Primitive('scalar_gt')
scalar_ge = Primitive('scalar_ge')
scalar_le = Primitive('scalar_le')
scalar_lt = Primitive('scalar_lt')
scalar_eq = Primitive('scalar_eq')
scalar_ne = Primitive('scalar_ne')
scalar_uadd = Primitive(_constants.kScalarUadd)
scalar_usub = Primitive(_constants.kScalarUsub)
scalar_mod = Primitive(_constants.kScalarMod)
string_eq = Primitive('string_equal')
string_concat = Primitive('string_concat')
bool_not = Primitive("bool_not")
bool_or = Primitive("bool_or")
bool_and = Primitive("bool_and")
bool_eq = Primitive("bool_eq")
logical_and = P.LogicalAnd()
logical_or = P.LogicalOr()
logical_not = P.LogicalNot()
array_to_scalar = Primitive('array_to_scalar')
is_ = Primitive("is_")
is_not = Primitive("is_not")
in_dict = Primitive("in_dict")
not_in_dict = Primitive("not_in_dict")
mixed_precision_cast = Primitive("mixed_precision_cast")
broadcast_gradient_args = Primitive('BroadcastGradientArgs')
array_reduce = Primitive('array_reduce')
zeros_like = P.ZerosLike()
distribute = Primitive('distribute')
embed = Primitive('embed')
ref_to_embed = _grad_ops.RefToEmbed()
env_setitem = Primitive('env_setitem')
env_getitem = Primitive('env_getitem')
env_add = Primitive('env_add')
J = Primitive('J')
switch = Primitive('Switch')
switch_layer = Primitive('switch_layer')
# for sum bprop
reduced_shape = Primitive("reduced_shape")
# shape_mul:input mush be shape multiply elemts in tuple(shape)
shape_mul = Primitive("shape_mul")
# a primitive to compare between tuple.
stop_gradient = Primitive("stop_gradient")

make_row_tensor = Primitive('MakeRowTensor')
row_tensor_get_values = Primitive('RowTensorGetValues')
row_tensor_get_indices = Primitive('RowTensorGetIndices')
row_tensor_get_dense_shape = Primitive('RowTensorGetDenseShape')
row_tensor_add = Primitive('RowTensorAdd')

make_sparse_tensor = Primitive('MakeSparseTensor')
sparse_tensor_get_values = Primitive('SparseTensorGetValues')
sparse_tensor_get_indices = Primitive('SparseTensorGetIndices')
sparse_tensor_get_dense_shape = Primitive('SparseTensorGetDenseShape')

tensor_operator_registry.register('__add__', tensor_add)
tensor_operator_registry.register('__sub__', tensor_sub)
tensor_operator_registry.register('__mul__', tensor_mul)
tensor_operator_registry.register('__truediv__', tensor_div)
tensor_operator_registry.register('__mod__', tensor_mod)
tensor_operator_registry.register('__pow__', tensor_pow)
tensor_operator_registry.register('__floordiv__', tensor_floordiv)
tensor_operator_registry.register('all', P.ReduceAll)
tensor_operator_registry.register('any', P.ReduceAny)
tensor_operator_registry.register('abs', P.Abs)
tensor_operator_registry.register('mean', P.ReduceMean)
tensor_operator_registry.register('reshape', P.Reshape)
tensor_operator_registry.register('transpose', P.Transpose)
tensor_operator_registry.register('broadcast_to', P.BroadcastTo)
# ms cannot support Tensor(True) compare
tensor_operator_registry.register('__eq__', equal)
tensor_operator_registry.register('__ne__', not_equal)
tensor_operator_registry.register('__neg__', neg_tensor)
tensor_operator_registry.register('__lt__', tensor_lt)
tensor_operator_registry.register('__le__', tensor_le)
tensor_operator_registry.register('__gt__', tensor_gt)
tensor_operator_registry.register('__ge__', tensor_ge)
tensor_operator_registry.register('__logical_not__', logical_not)
tensor_operator_registry.register('shape', shape)
tensor_operator_registry.register('squeeze', squeeze)
# support GE backend for no compare operators
tensor_operator_registry.register('cast', cast)

__all__ = [name for name in dir() if name[0] != "_"]
__all__.remove('Primitive')
