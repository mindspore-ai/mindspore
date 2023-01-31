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
""" opt_test """
import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import Primitive
from mindspore.ops import _constants as Constants
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _grad_ops as G

# pylint: disable=unused-variable

# opt test data, not for running
# pylint: disable=unused-argument
# pylint: disable=redefined-outer-name

scalar_add = F.scalar_add
scalar_mul = F.scalar_mul
tuple_getitem = Primitive(Constants.kTupleGetItem)
switch = Primitive('Switch')


def test_sexp_conversion():
    """ test_sexp_conversion """
    return scalar_mul(10, scalar_add(5, 4))


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_add_zero(tag):
    """ test_add_zero """
    fns = FnDict()

    @fns
    def before_1(x):
        return scalar_add(x, 0)

    @fns
    def before_2(x):
        return scalar_add(scalar_add(x, 0), 0)

    @fns
    def after(x):
        return x

    return fns[tag]


def test_elim_r(tag):
    """
    Feature: optimizer.
    Description: test elimi R.
    Expectation: run case with no exception.
    """
    R = Primitive('R')

    fns = FnDict()

    @fns
    def before_1(x):
        return R(x)

    @fns
    def after(x):
        return x

    return fns[tag]


def test_idempotent(tag):
    """ test_idempotent """
    P = Primitive('P')
    R = Primitive('R')

    fns = FnDict()

    @fns
    def before_1(x):
        return P(P(x))

    @fns
    def before_2(x):
        return P(P(P(P(P(x)))))

    @fns
    def after(x):
        return P(x)

    return fns[tag]


def test_constant_variable(tag):
    """ test_constant_variable """
    P = Primitive('P')
    Q = Primitive('Q')

    fns = FnDict()

    @fns
    def before_1(x):
        return Q(15) + Q(x)

    @fns
    def after(x):
        return P(15) + Q(x)

    return fns[tag]


def cost(x):
    """ cost """
    return x * 10


J = Primitive('J')


def test_expandJ(x):
    """ test_expandJ """
    return J(cost)(x)


def test_elim_jinv_j(tag):
    """ test_elim_jinv_j """
    J = Primitive('J')
    Jinv = Primitive('Jinv')

    fns = FnDict()

    @fns
    def before_1(x):
        return J(Jinv(x))

    @fns
    def before_2(x):
        return Jinv(J(x))

    @fns
    def after(x):
        return x

    return fns[tag]


def test_simplify_always_true_false(tag):
    """ test_simplify_always_true_false """
    fns = FnDict()

    @fns
    def before_1(x, y):
        return switch(True, x, y)

    @fns
    def before_2(x, y):
        return switch(False, y, x)

    @fns
    def after(x, y):
        return x

    return fns[tag]


def test_inline(tag):
    """ test_inline """
    fns = FnDict()

    @fns
    def before(x, y):
        def fn1(x1):
            return x1

        return fn1(x)

    @fns
    def after(x, y):
        return x

    return fns[tag]


def test_inline_successively(tag):
    """ test_inline_successively """
    fns = FnDict()

    def one(x):
        return x + 1

    def two(x):
        return one(x + 2)

    def three(x):
        return two(x + 3)

    @fns
    def before(x):
        return three(x)

    @fns
    def after(x):
        return x + 3 + 2 + 1

    return fns[tag]


def test_inline_closure(tag):
    """ test_inline_closure """
    fns = FnDict()

    @fns
    def before(x, y, z):
        c = z * z

        def f(x):
            return x + c

        return f(x * y)

    @fns
    def after(x, y, z):
        c = z * z
        return x * y + c

    return fns[tag]


def test_inline_deep_closure(tag):
    """ test_inline_deep_closure """
    fns = FnDict()

    def f(x):
        w = x * x

        def g():
            def h():
                return w

            return h()

        return g

    @fns
    def before(x, y):
        return f(x)() - f(y)()

    @fns
    def after(x, y):
        w1 = x * x
        w2 = y * y
        return w1 - w2

    return fns[tag]


def test_inline_new_closure(tag):
    """ test_inline_new_closure """
    fns = FnDict()

    def q(x):
        return x * x

    def f(x):
        def g():
            return q(x)

        return g

    @fns
    def before(x):
        return f(x)

    @fns
    def after(x):
        def g():
            return x * x

        return g

    return fns[tag]


def test_inline_recursive_direct(tag):
    """ test_inline_recursive_direct """
    fns = FnDict()

    @fns
    def before1(x):
        return before1(x - 1)

    @fns
    def before2(x):
        def helper1(x):
            return before2(x - 1)

        def helper2(x):
            return before1(x - 1)

        return helper1(x)

    return fns[tag]


def test_inline_recursive(tag):
    """ test_inline_recursive """
    fns = FnDict()

    @fns
    def before(x):
        if x <= 0:
            return x
        return before(x - 1)

    return fns[tag]


def test_inline_while(tag):
    """ test_inline_while """
    fns = FnDict()

    @fns
    def before(x):
        rval = x
        while rval < 100:
            rval = rval * rval
        return rval

    return fns[tag]


def test_cse(tag):
    """ test_cse """
    fns = FnDict()
    scalar_div = F.scalar_div

    @fns
    def test_f1(x, y):
        a = scalar_add(x, y)
        b = scalar_add(x, y)
        c = scalar_mul(a, b)
        return c

    @fns
    def test_f2(x, y):
        a = scalar_add(x, y)
        b = scalar_add(scalar_mul(a, y), scalar_div(a, x))
        c = scalar_add(scalar_mul(a, y), scalar_div(scalar_add(x, y), x))
        d = scalar_add(b, c)
        return d

    return fns[tag]


def test_no_grad(tag):
    """
    Feature: test no grad input net.
    Description: test no grad input net.
    Expectation: No exception.
    """
    fns = FnDict()
    mul = Primitive('Mul')
    make_tuple = Primitive('MakeTuple')

    @fns
    def test_f1(x, y):
        x1 = mul(x, 2)
        y1 = mul(y, 2)
        return make_tuple(x1, y1)

    return fns[tag]


def test_arithmetic(tag):
    """ test_arithmetic """
    fns = FnDict()
    identity = Primitive('identity')

    @fns
    def multiply_by_zero_l(x):
        return scalar_mul(x, 0)

    @fns
    def multiply_by_zero_r(x):
        return scalar_mul(0, x)

    @fns
    def after_0(x):
        return 0

    @fns
    def multiply_by_one_l(x):
        return scalar_mul(x, 1)

    @fns
    def multiply_by_one_r(x):
        return scalar_mul(1, x)

    @fns
    def add_zero_l(x):
        return scalar_add(x, 0)

    @fns
    def add_zero_r(x):
        return scalar_add(0, x)

    @fns
    def elim_identity(x):
        return identity(x)

    @fns
    def after(x):
        return x

    return fns[tag]


def test_elim_cast_same_dtype(tag):
    """ test_elim_cast_same_dtype """
    fns = FnDict()
    cast = P.Cast()

    @fns
    def fp32_cast_fp32(x, y):
        return cast(x, y)

    @fns
    def after(x, y):
        return x

    return fns[tag]


def elim_reshape_same_shape(tag):
    """ elim_reshape_same_shape """
    fns = FnDict()
    reshape = P.Reshape()
    shape = (2, 3)

    @fns
    def reshape_to_2_3(x):
        return reshape(x, shape)

    @fns
    def after(x):
        return x

    return fns[tag]


def elim_two_reshape(tag):
    """ elim_two_reshape """
    fns = FnDict()
    reshape = P.Reshape()
    shape = (2, 3)
    shape_2 = (3, 2)

    @fns
    def before(x):
        return reshape(reshape(x, shape_2), shape)

    @fns
    def after(x):
        return reshape(x, shape)

    return fns[tag]


def test_elim_transpose(tag):
    """ test_elim_transpose """
    fns = FnDict()
    transpose = P.Transpose()
    perm = (0, 1, 2)

    @fns
    def before(x):
        return transpose(x, perm)

    @fns
    def after(x):
        return x

    return fns[tag]


def test_elim_depend_value(tag):
    """ test_elim_depend_value """
    fns = FnDict()
    depend = P.Depend()

    @fns
    def before(x):
        return depend(x, None)

    @fns
    def after(x):
        return x

    return fns[tag]


def test_elim_tile_multiply_one(tag):
    """ test_elim_tile_multiply_one """
    fns = FnDict()
    tile = P.Tile()
    all_one = (1, 1, 1)

    @fns
    def before(x):
        return tile(x, all_one)

    @fns
    def after(x):
        return x

    return fns[tag]


def test_elim_reduce_mean_shape_one(tag):
    """ test_elim_reduce_mean_shape_one """
    fns = FnDict()
    reduce_mean = P.ReduceMean()

    @fns
    def before(x, y):
        return reduce_mean(x, 0)

    @fns
    def after(x, y):
        return x

    return fns[tag]


def test_elim_all_shape_one(tag):
    """ test_elim_all_shape_one """
    fns = FnDict()
    all_ = P.ReduceAll()

    @fns
    def before(x, y):
        return all_(x, 0)

    @fns
    def after(x, y):
        return x

    return fns[tag]


def test_elim_sum_shape_one(tag):
    """ test_elim_sum_shape_one """
    fns = FnDict()
    sum_ = P.ReduceSum()

    @fns
    def before(x, y):
        return sum_(x, 0)

    @fns
    def after(x, y):
        return x

    return fns[tag]


def test_tuple_getitem(tag):
    """ test_tuple_getitem """
    fns = FnDict()
    make_tuple = Primitive('MakeTuple')

    @fns
    def make_get_0(x, y):
        return tuple_getitem(make_tuple(x, y), 0)

    @fns
    def make_get_1(x, y):
        return tuple_getitem(make_tuple(x, y), 1)

    @fns
    def after_0(x, y):
        return x

    @fns
    def after_1(x, y):
        return y

    return fns[tag]


def test_tuple_setitem(tag):
    """ test_tuple_setitem """
    fns = FnDict()
    make_tuple = Primitive('MakeTuple')
    tuple_setitem = Primitive('tuple_setitem')

    @fns
    def before_0(x, y, z):
        return tuple_setitem(make_tuple(x, y), 0, z)

    @fns
    def before_1(x, y, z):
        return tuple_setitem(make_tuple(x, y), 1, z)

    @fns
    def after_0(x, y, z):
        return make_tuple(z, y)

    @fns
    def after_1(x, y, z):
        return make_tuple(x, z)

    return fns[tag]


def test_tuple_get_set_item(tag):
    """ test_tuple_get_set_item """
    fns = FnDict()
    tuple_setitem = Primitive('tuple_setitem')

    @fns
    def before_0(t, x):
        return tuple_getitem(tuple_setitem(t, 0, x), 0)

    @fns
    def after_0(t, x):
        return x

    @fns
    def before_1(t, x):
        return tuple_getitem(tuple_setitem(t, 0, x), 1)

    @fns
    def after_1(t, x):
        return tuple_getitem(t, 1)

    return fns[tag]


def test_partial(tag):
    """ test_partial """
    fns = FnDict()
    partail = P.Partial()

    def f(x, y):
        return scalar_add(x, y)

    @fns
    def before(x, y):
        return partail(f, x)(y)

    @fns
    def after(x, y):
        return f(x, y)

    return fns[tag]


def test_replace_applicator(tag):
    """ test_replace_applicator """
    fns = FnDict()
    partail = P.Partial()

    def app1(x, y):
        return scalar_add(x, y)

    def app2(x, y):
        return app1(x, y)

    def app3(x, y):
        return scalar_add(y, x)

    @fns
    def before1(x, y):
        return app1(x, y)

    @fns
    def before2(x, y):
        return app2(x, y)

    @fns
    def before3(x, y):
        return app3(x, y)

    @fns
    def after(x, y):
        return scalar_add(x, y)

    return fns[tag]


def test_specialize_on_graph_arguments(tag):
    """ test_specialize_on_graph_arguments """
    fns = FnDict()
    f1 = Primitive('f1')
    f2 = Primitive('f2')

    @fns
    def before(x, y):
        def helper(f, x, g, y):
            return scalar_add(f(x), g(y))

        return helper(f1, x, f2, y)

    @fns
    def after(x, y):
        def helper(x, y):
            return scalar_add(f1(x), f2(y))

        return helper(x, y)

    return fns[tag]


def test_incorporate_getitem(tag):
    """ test_incorporate_getitem """
    fns = FnDict()
    f1 = Primitive('f1')
    f2 = Primitive('f2')

    @fns
    def before1(x, y):
        def fn(x, y):
            return f1(x, y), f2(x, y)

        return tuple_getitem(fn(x, y), 0)

    @fns
    def after1(x, y):
        def fn(x, y):
            return f1(x, y)

        return fn(x, y)

    @fns
    def before2(x, y):
        def fn(x, y):
            return x

        return tuple_getitem(fn(x, y), 0)

    @fns
    def after2(x, y):
        def fn(x, y):
            return tuple_getitem(x, 0)

        return fn(x, y)

    return fns[tag]


def test_incorporate_getitem_through_switch(tag):
    """ test_incorporate_getitem_through_switch """
    fns = FnDict()
    scalar_gt = F.scalar_gt

    @fns
    def before(x, y):
        def f1(x, y):
            return x, y

        def f2(x, y):
            return y, x

        return tuple_getitem(
            switch(scalar_gt(x, 0), f1, f2)(x, y),
            0)

    @fns
    def after(x, y):
        def f1(x, y):
            return x

        def f2(x, y):
            return y

        return switch(scalar_gt(x, 0), f1, f2)(x, y)

    return fns[tag]


def test_incorporate_call(tag):
    """ test_incorporate_call """
    fns = FnDict()
    f1 = Primitive('f1')

    @fns
    def before(x, y):
        def fn(q):
            def subf(z):
                return f1(q, z)

            return subf

        return fn(x)(y)

    @fns
    def after(x, y):
        def fn(q, y):
            def subf(z):
                return f1(q, z)

            return subf(y)

        return fn(x, y)

    return fns[tag]


def test_incorporate_call_through_switch(tag):
    """ test_incorporate_call_through_switch """
    fns = FnDict()
    f1 = Primitive('f1')
    f2 = Primitive('f2')
    scalar_gt = F.scalar_gt
    identity = Primitive('identity')

    @fns
    def before(x, y, z):
        def f1g():
            return f1

        def f2g():
            return f2

        def fn():
            return switch(scalar_gt(x, 0), f1g, f2g)()

        return fn()(y, z)

    @fns
    def after(x, y, z):
        def fn(y, z):
            def tb(y, z):
                return f1(y, z)

            def fb(y, z):
                return f2(y, z)

            return switch(scalar_gt(x, 0), tb, fb)(y, z)

        return fn(y, z)

    return fns[tag]


def test_float_tuple_getitem_through_switch(tag):
    """ test_float_tuple_getitem_through_switch """
    fns = FnDict()
    scalar_gt = F.scalar_gt

    @fns
    def before(x, y):
        return tuple_getitem(switch(scalar_gt(x, 0), x, y), 0)

    @fns
    def after(x, y):
        return switch(scalar_gt(x, 0), tuple_getitem(x, 0), tuple_getitem(y, 0))

    return fns[tag]


def test_merge_addn(tag):
    """ test_merge_addn """
    fns = FnDict()
    addn = P.AddN()

    @fns
    def before(x, y, z, a):
        return addn((addn((a, x, y)), z))

    @fns
    def after(x, y, z, a):
        return addn((a, x, y, z))

    return fns[tag]


def test_addn_zero(tag):
    """ test_addn_zero """
    fns = FnDict()
    addn = P.AddN()
    zero_tensor = Primitive('ZerosLike')

    @fns
    def before_1(x, y, z, a):
        return addn((a, zero_tensor(x), zero_tensor(y), z))

    @fns
    def after(x, y, z, a):
        return addn((a, z))

    @fns
    def before_2(x, y, z, a):
        return addn((a, zero_tensor(x), z, zero_tensor(y)))

    @fns
    def before_3(x, y, z, a):
        return addn((zero_tensor(x), a, z, zero_tensor(y)))

    @fns
    def before_4(x, y, z, a):
        return addn((a, z))

    return fns[tag]


def test_convert_switch_ops(tag):
    fns = FnDict()
    ge_switch = Primitive('GeSwitch')
    merge = Primitive('Merge')
    add = F.scalar_add
    neg = Primitive('Neg')
    tuple_getitem = Primitive(Constants.kTupleGetItem)
    make_tuple = Primitive('MakeTuple')

    @fns
    def before(cond, x, y):
        if cond:
            z = add(x, y)
        else:
            z = neg(y)
        return z

    @fns
    def after(cond, x, y):
        sw1 = ge_switch(x, cond)
        sw2 = ge_switch(y, cond)
        sw3 = ge_switch(y, cond)
        sw1_t = tuple_getitem(sw1, 1)
        sw2_t = tuple_getitem(sw2, 1)
        sw3_f = tuple_getitem(sw3, 0)
        t_res = add(sw1_t, sw2_t)
        f_res = neg(sw3_f)
        tup = make_tuple(t_res, f_res)
        merge_res = merge(tup)
        res = tuple_getitem(merge_res, 0)
        return res

    return fns[tag]


def test_minmax_grad(tag):
    """ test_minmax_grad """
    fns = FnDict()
    min_grad = G.MinimumGrad()

    @fns
    def before_11(x, y, dout):
        return tuple_getitem(min_grad(x, y, dout), 0)

    @fns
    def before_12(x, y, dout):
        return tuple_getitem(min_grad(x, y, dout), 1)

    @fns
    def before_2(x, y, dout):
        a = min_grad(x, y, dout)
        return tuple_getitem(a, 0), tuple_getitem(a, 1)

    max_grad = G.MaximumGrad()

    @fns
    def before_31(x, y, dout):
        return tuple_getitem(max_grad(x, y, dout), 0)

    @fns
    def before_32(x, y, dout):
        return tuple_getitem(max_grad(x, y, dout), 1)

    @fns
    def before_4(x, y, dout):
        a = max_grad(x, y, dout)
        return tuple_getitem(a, 0), tuple_getitem(a, 1)

    return fns[tag]


def test_reducesum_one(tag):
    """ test_reducesum_one """
    fns = FnDict()
    reduce_sum_true = P.ReduceSum(keep_dims=True)
    reduce_sum_false = P.ReduceSum(keep_dims=False)
    axis = (2, 3)
    reshape = P.Reshape()
    shape1 = (3, 2, 2)
    shape2 = (3, 2)

    @fns
    def before_1(x):
        return reduce_sum_true(x, 3)

    @fns
    def before_2(x):
        return reduce_sum_true(x, axis)

    @fns
    def before_3(x):
        return reduce_sum_false(x, 3)

    @fns
    def before_4(x):
        return reduce_sum_false(x, axis)

    @fns
    def after_1(x):
        return x

    @fns
    def after_2(x):
        return reshape(x, shape1)

    @fns
    def after_3(x):
        return reshape(x, shape2)

    return fns[tag]


def test_print_tuple_wrapper(tag):
    fns = FnDict()
    print_ = Primitive('Print')
    make_tuple = Primitive('MakeTuple')

    @fns
    def before1(x, y):
        return print_(x, y)

    @fns
    def after1(x, y):
        return print_(make_tuple(x, y))

    @fns
    def before2(x, y, z):
        return print_(x, make_tuple(y, z))

    @fns
    def after2(x, y, z):
        return print_(make_tuple(x, make_tuple(y, z)))

    @fns
    def before3(x, y, z):
        return print_(make_tuple(x, y, z))

    return fns[tag]


# pylint: disable=unnecessary-semicolon
def test_constant_duplicate_mul(tag):
    fns = FnDict()
    Mul = Primitive('Mul')
    Sqrt = Primitive('Sqrt')

    x = Tensor(np.array([[2, 2], [2, 3]]).astype('float32'))
    tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
    tensor2 = Tensor(np.array([[2.2, 3.1], [3.2, 4.2]]).astype('float32'))

    @fns
    def beforell():
        return Mul(tensor1, Mul(tensor2, Sqrt(x)))

    @fns
    def beforelr():
        return Mul(tensor1, Mul(Sqrt(x), tensor2))

    @fns
    def beforerl():
        return Mul(Mul(Sqrt(x), tensor2), tensor1)

    @fns
    def beforerr():
        return Mul(Mul(Sqrt(x), tensor2), tensor1)

    @fns
    def after():
        return Mul(Sqrt(x), Mul(tensor1, tensor2))

    return fns[tag]


def test_adjust_allreduce_mul_add(tag):
    fns = FnDict()
    Mul = Primitive('Mul')
    AddN = Primitive('AddN')
    AllReduce = Primitive('AllReduce')

    x = Tensor(np.ones(shape=(64, 32)).astype(np.float32))
    y = Tensor(np.ones(shape=(64, 32)).astype(np.float32))
    z = Tensor(np.ones(shape=(64, 32)).astype(np.float32))

    @fns
    def beforell():
        return AddN((z, Mul(y, AllReduce(x))))

    @fns
    def beforelr():
        return AddN((z, Mul(AllReduce(x), y)))

    @fns
    def beforerl():
        return AddN((Mul(y, AllReduce(x)), z))

    @fns
    def beforerr():
        return AddN((Mul(AllReduce(x), y), z))

    @fns
    def after1():
        return Mul(AllReduce(AddN((z, x))), y)

    @fns
    def before2r():
        return AddN((Mul(AllReduce(x), y), Mul(z, z)))

    @fns
    def before2l():
        return AddN((Mul(z, z), Mul(AllReduce(x), y)))

    @fns
    def after2():
        return Mul(AllReduce(AddN((Mul(z, z), x))), y)

    return fns[tag]


def test_row_tensor(tag):
    """ test_add_zero """
    fns = FnDict()
    make_row_tensor = Primitive('MakeRowTensor')
    row_tensor_get_values = Primitive('RowTensorGetValues')
    row_tensor_get_indices = Primitive('RowTensorGetIndices')
    row_tensor_get_dense_shape = Primitive('RowTensorGetDenseShape')

    @fns
    def before_get_indices(x, y, z):
        return row_tensor_get_indices(make_row_tensor(x, y, z))

    @fns
    def after_get_indices(x, y, z):
        return x

    @fns
    def before_get_values(x, y, z):
        return row_tensor_get_values(make_row_tensor(x, y, z))

    @fns
    def after_get_values(x, y, z):
        return y

    @fns
    def before_get_dense_shape(x, y, z):
        return row_tensor_get_dense_shape(make_row_tensor(x, y, z))

    @fns
    def after_get_dense_shape(x, y, z):
        return z

    return fns[tag]


def test_sparse_tensor(tag):
    """ test_add_zero """
    fns = FnDict()
    make_sparse_tensor = Primitive('MakeCOOTensor')
    sparse_tensor_get_values = Primitive('COOTensorGetValues')
    sparse_tensor_get_indices = Primitive('COOTensorGetIndices')
    sparse_tensor_get_dense_shape = Primitive('COOTensorGetDenseShape')

    @fns
    def before_get_indices(x, y, z):
        return sparse_tensor_get_indices(make_sparse_tensor(x, y, z))

    @fns
    def after_get_indices(x, y, z):
        return x

    @fns
    def before_get_values(x, y, z):
        return sparse_tensor_get_values(make_sparse_tensor(x, y, z))

    @fns
    def after_get_values(x, y, z):
        return y

    @fns
    def before_get_dense_shape(x, y, z):
        return sparse_tensor_get_dense_shape(make_sparse_tensor(x, y, z))

    @fns
    def after_get_dense_shape(x, y, z):
        return z

    return fns[tag]


# Test ut for file: call_graph_tuple_transform.h.
def test_tuple_flatten(tag):
    """
    Feature: optimizer.
    Description: test cases for pass: graph_tuple_transform.
    Expectation: the tuple args and parameters are successfully flattened by the pass.
    """
    fns = FnDict()
    w = Tensor(np.random.randn(64, 3, 7, 7).astype(np.float32))
    x = Tensor(np.random.randn(32, 3, 224, 224).astype(np.float32))
    y = Tensor(np.random.randn(32, 3, 224, 224).astype(np.float32))

    p = Tensor(3, mstype.float32)

    out_channel = 64
    kernel_size = 7
    conv = P.Conv2D(out_channel,
                    kernel_size,
                    mode=1,
                    pad_mode="valid",
                    pad=0,
                    stride=1,
                    dilation=1,
                    group=1)
    pow_ops = P.Pow()

    @fns
    def test_flatten_switch_partial_arg():
        def called_graph_with_tuple(tuple_x, tuple_y):
            return conv(F.tuple_getitem(tuple_x, 0), F.tuple_getitem(tuple_x, 1)) + conv(F.tuple_getitem(tuple_y, 0),
                                                                                         F.tuple_getitem(tuple_y, 1))

        # Add tuple args in partial args.
        func1 = F.partial(called_graph_with_tuple, (pow_ops(x, p), pow_ops(w, p)))
        func2 = F.partial(called_graph_with_tuple, (pow_ops(x, p), pow_ops(w, p)))
        cond = x < y

        switch_node = F.switch(cond, func1, func2)
        # Add tuple args in call args.
        return switch_node((pow_ops(x, p), pow_ops(w, p)))

    index = Tensor(1, mstype.int32)

    @fns
    def test_flatten_switch_layer_partial_arg():
        def called_graph_with_tuple(tuple_x):
            return conv(F.tuple_getitem(tuple_x, 0), F.tuple_getitem(tuple_x, 1))

        def called_graph_no_tuple(param1, param2):
            return conv(param1, param2)

        # Add tuple args in partial
        func1 = F.partial(called_graph_with_tuple, (pow_ops(x, p), pow_ops(w, p)))
        func2 = F.partial(called_graph_with_tuple, (pow_ops(x, p), pow_ops(w, p)))
        # Add tensor args in partial
        func3 = F.partial(called_graph_no_tuple, pow_ops(x, p), pow_ops(w, p))
        switch_node = F.switch_layer(pow_ops(index, index), (func1, func2, func3))
        return switch_node()

    @fns
    def test_flatten_simple_call_tuple_in_tuple_arg():
        def called_graph_with_tuple(tuple_x, tuple_tuple_y, tensor_z):
            result1 = conv(F.tuple_getitem(tuple_x, 0), F.tuple_getitem(tuple_x, 1))
            tuple_0 = F.tuple_getitem(tuple_tuple_y, 0)
            result2 = conv(F.tuple_getitem(tuple_0, 0), F.tuple_getitem(tuple_0, 1))
            tensor_1 = F.tuple_getitem(tuple_tuple_y, 1)
            result3 = conv(tensor_1, tensor_z)
            return result1 + result2 + result3

        # Tuple arg.
        tuple_x_arg = (pow_ops(x, p), pow_ops(w, p))
        # TupleTuple arg.
        tuple_0_arg = (pow_ops(x, p), pow_ops(w, p))
        tensor_1_arg = pow_ops(x, p)
        tuple_tuple_y_arg = (tuple_0_arg, tensor_1_arg)
        # TensorArg
        tensor_z_arg = pow_ops(w, p)
        return called_graph_with_tuple(tuple_x_arg, tuple_tuple_y_arg, tensor_z_arg)

    return fns[tag]
