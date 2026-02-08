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
import sys
import numpy as np
import mindspore as ms


def fn_keyword_arguments():
    @ms.jit
    @ms.enable_dynamic(y=ms.Tensor(shape=[None, None], dtype=ms.float32),
                       z=ms.Tensor(shape=[2, None], dtype=ms.float32))
    def fn(x, y, z):
        return x + 1, y + 1, z + 1

    x1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    y2 = ms.Tensor(np.random.randn(3, 2), ms.float32)
    y3 = ms.Tensor(np.random.randn(4, 4), ms.float32)

    z1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    z2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    fn(x1, y1, z1)
    fn(x2, y2, z2)
    fn(x3, y3, z3)


def fn_varargs_kwargs():
    @ms.enable_dynamic(x=ms.Tensor(shape=None, dtype=ms.float32),
                       a=ms.Tensor(shape=[2, None], dtype=ms.float32),
                       y=ms.Tensor(shape=None, dtype=ms.float32))
    @ms.jit
    def fn(x, y, a, b, *args, **kwargs):
        return x + 1, y + 1, a + b, args[0] + args[1]

    x1 = ms.Tensor(np.random.randn(1, 1), ms.float32)
    x2 = ms.Tensor(np.random.randn(1, 2), ms.float32)
    x3 = ms.Tensor(np.random.randn(1, 2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    y2 = ms.Tensor(np.random.randn(1), ms.float32)
    y3 = ms.Tensor(np.random.randn(1, 2, 3, 4), ms.float32)

    a1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    a2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    a3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    t1 = ms.Tensor(np.random.randn(3, 3), ms.float32)
    t2 = ms.Tensor(np.random.randn(3, 3), ms.float32)
    t3 = ms.Tensor(np.random.randn(3, 3), ms.float32)

    fn(x1, y1, a1, 1, t1, t1)
    fn(x2, y2, a2, 1, t2, t2)
    fn(x3, y3, a3, 1, t3, t3)


def fn_decorator_callable():
    def _fn(a, b, c):
        return a + 1, b + 1, c + 1
    fn = ms.enable_dynamic(a=ms.Tensor(shape=None, dtype=ms.float32),
                           b=ms.Tensor(shape=None, dtype=ms.float32),
                           c=ms.Tensor(shape=None, dtype=ms.float32))(ms.jit(_fn))

    x1 = ms.Tensor(np.random.randn(1, 1), ms.float32)
    x2 = ms.Tensor(np.random.randn(1, 2), ms.float32)
    x3 = ms.Tensor(np.random.randn(1, 2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    y2 = ms.Tensor(np.random.randn(1), ms.float32)
    y3 = ms.Tensor(np.random.randn(1, 2, 3, 4), ms.float32)

    z1 = ms.Tensor(np.random.randn(2), ms.float32)
    z2 = ms.Tensor(np.random.randn(3), ms.float32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    fn(x1, y1, z1)
    fn(x2, y2, z2)
    fn(x3, y3, z3)


def fn_list_arguments():
    @ms.jit
    @ms.enable_dynamic(y=[ms.Tensor(shape=[None, 1], dtype=ms.float32), ms.Tensor(shape=[2, None], dtype=ms.float32)])
    def fn(x, y):
        return x + 1, y[0] + 1, y[1] + 1

    x1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    y2 = ms.Tensor(np.random.randn(3, 1), ms.float32)
    y3 = ms.Tensor(np.random.randn(4, 1), ms.float32)

    z1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    z2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    fn(x1, ms.mutable([y1, z1]))
    fn(x2, ms.mutable([y2, z2]))
    fn(x3, ms.mutable([y3, z3]))


def fn_tuple_arguments():
    @ms.jit
    @ms.enable_dynamic(y=(ms.Tensor(shape=[None, 1], dtype=ms.float32), ms.Tensor(shape=[2, None], dtype=ms.float32)))
    def fn(x, y):
        return x + 1, y[1] + 1, y[0] *2

    x1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    y2 = ms.Tensor(np.random.randn(3, 1), ms.float32)
    y3 = ms.Tensor(np.random.randn(4, 1), ms.float32)

    z1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    z2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    fn(x1, ms.mutable([y1, z1]))
    fn(x2, ms.mutable([y2, z2]))
    fn(x3, ms.mutable([y3, z3]))


def fn_nested_tuple_arguments():
    d1 = ms.Tensor(shape=[None, 4], dtype=ms.float32)
    d2 = ms.Tensor(shape=[3, None], dtype=ms.float32)
    ds = [d1, (d1, d2)]

    @ms.jit
    @ms.enable_dynamic(x=ds)
    def fn(x):
        return x[0] * 2, x[1][1] * 3

    x_shapes = [(1, 4), (2, 4), (3, 4)]
    y_shapes = [(3, 4), (3, 2), (3, 3)]
    for i in range(3):
        x = ms.Tensor(np.random.randn(*x_shapes[i]), ms.float32)
        y = ms.Tensor(np.random.randn(*y_shapes[i]), ms.float32)
        fn(ms.mutable([x, (x, y)]))


def fn_with_dynamic():
    @ms.jit(dynamic=1)
    @ms.enable_dynamic(y=ms.Tensor(shape=None, dtype=ms.float32),
                       x=ms.Tensor(shape=[2, None], dtype=ms.float32))
    def fn(x, y, z):
        return x + 1, y + 1, z + 1

    x1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x4 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    y2 = ms.Tensor(np.random.randn(3, 4, 5), ms.float32)
    y3 = ms.Tensor(np.random.randn(4, 5, 6, 7), ms.float32)
    y4 = ms.Tensor(np.random.randn(5, 6, 7, 8, 9), ms.float32)

    z1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    z2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    z4 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    fn(x1, y1, z1)
    fn(x2, y2, z2)
    fn(x3, y3, z3)
    fn(x4, y4, z4)


def fn_with_dynamic_and_tuple_arguments():
    @ms.jit(dynamic=1)
    @ms.enable_dynamic(x=[ms.Tensor(shape=[2, None], dtype=ms.float32), ms.Tensor(shape=[2, 2], dtype=ms.float32)])
    def fn(x, y):
        return x[0] + 1, y + 1

    x1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x4 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    y2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    y3 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    y4 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    list1 = [x1, x2]
    list2 = [x2, x2]
    list3 = [x3, x2]
    list4 = [x4, x2]

    fn(ms.mutable(list1), y1)
    fn(ms.mutable(list2), y2)
    fn(ms.mutable(list3), y3)
    fn(ms.mutable(list4), y4)


if __name__ == "__main__":
    fn_name = sys.argv[1]
    if fn_name == "fn1":
        fn_keyword_arguments()
    elif fn_name == "fn2":
        fn_varargs_kwargs()
    elif fn_name == "fn3":
        fn_decorator_callable()
    elif fn_name == "fn4":
        fn_list_arguments()
    elif fn_name == "fn5":
        fn_tuple_arguments()
    elif fn_name == "fn6":
        fn_nested_tuple_arguments()
    elif fn_name == "fn7":
        fn_with_dynamic()
    elif fn_name == "fn8":
        fn_with_dynamic_and_tuple_arguments()
