# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
from mindspore.ops import Primitive

py_execute_need_cast = Primitive("PyExecute")
py_execute_do_not_cast = Primitive("PyExecute")
py_execute_need_cast.add_prim_attr("need_cast", True)
cast = Primitive('Cast')


class FnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        try:
            return self.fn_dict[name]
        except KeyError:
            return None


def insert_cast_for_py_execute(tag):
    """
    Feature: test pass for pyexecute cast insert pass
    Description: test pass is correct
    Expectation: No error
    """
    fns = FnDict()

    @fns
    def before(x, y):
        x = py_execute_need_cast("a(x)", "x", (x,))
        y = py_execute_do_not_cast("a(y)", "y", (y,))
        return x, y

    @fns
    def after(x, y):
        x = py_execute_need_cast("a(x)", "x", (x,))
        y = py_execute_do_not_cast("a(y)", "y", (y,))
        x = cast(x)
        return ((x, y),)

    return fns[tag]
