# Copyright 2023 Huawei Technologies Co., Ltd
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

py_interpret = Primitive("PyInterpret")
py_execute = Primitive("PyExecute")
make_tuple = Primitive("MakeTuple")
make_dict = Primitive("make_dict")


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def py_interpret_to_py_execute_test(tag):
    """ test_split_bn_fusion """
    fns = FnDict()

    @fns
    def before():
        global_key = make_tuple("g_a", "g_b")
        global_value = make_tuple(1, 2)
        local_key = make_tuple("g_a", "a", "b")
        local_value = make_tuple(3, 4, 5)
        global_dict = make_dict(global_key, global_value)
        local_dict = make_dict(local_key, local_value)
        output = py_interpret("func(g_a, g_b, a, b)", global_dict, local_dict)
        return output

    @fns
    def after():
        local_key = make_tuple("g_b", "g_a", "a", "b")
        local_value = make_tuple(2, 3, 4, 5)
        output = py_execute("func(g_a, g_b, a, b)", local_key, local_value)
        return output

    return fns[tag]
