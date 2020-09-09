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
# ===========================================================================
"""test split"""
import model


def graph_1():
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a = gb.tensor([1024, 16], "float32", name="a")
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("Abs", b, 'c')
        d = gb.emit("Abs", c, 'd')
        gb.emit("TensorAdd", [b, d], "e")
    return gb.get()[0]


def graph_2():
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a = gb.tensor([1024, 16], "float32", name="a")
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("Abs", b, 'c')
        d = gb.emit("ReduceSum", c, 'd', attrs={'reduce_axis': (1,)})
        gb.emit("Sqrt", d, 'e')
    return gb.get()[0]


def test_split_by_pattern():
    def _test(graph):
        print("***************** main graph ***************")
        print(graph)
        subgraphs = model.split(graph)
        for i, g in enumerate(subgraphs):
            print('------------- subgraph {} --------------'.format(i))
            print(g)
    _test(graph_2())


if __name__ == '__main__':
    test_split_by_pattern()
