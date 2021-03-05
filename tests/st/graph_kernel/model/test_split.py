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
"""Test split"""
import model
from model import model as estimate
from model import graph_split as split


def get_nodes(sp, ops):
    """Get nodes"""
    if isinstance(ops[0], str):
        new_ops = []
        for t in ops:
            for op in sp.graph.ops:
                if op.output.name == t:
                    new_ops.append(op)
                    break
            else:
                print("ERROR: not found op: ", t)
        ops = new_ops
    return [sp.nodes[sp.graph.ops.index(op)] for op in ops]


def first_connected(sp, space):
    for cand in space:
        nodes = [sp.nodes[i] for i in cand[0]]
        graphs = sp.resolve_connnected_graphs(nodes)
        if len(graphs) != 1:
            print("connect check failed: ", nodes)
            return False
    return True


def split_format(sp, cand):
    names = []
    for ids in cand:
        ops = []
        for i in ids:
            ops.append(sp.graph.ops[i].output.name)
        names.append(','.join(ops))
    return '|'.join(names)


def graph_1():
    ''' ring, no succ_dep, no prev '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a = gb.tensor([10240, 16], "float32", name="a")
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("Abs", b, 'c')
        d = gb.emit("Abs", c, 'd')
        gb.emit('Add', [b, d], 'e')
    return gb.get()[0]


def graph_2():
    ''' ring, succ_dep, no prev '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([10240, 16], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("Abs", a, 'c')
        d = gb.emit("Abs", b, 'd')
        e = gb.emit('Add', [c, d], 'e')
        gb.emit("Abs", e, 'f')
    return gb.get()[0]


def graph_3():
    ''' no ring, 1 sibling node '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([10240, 16], "float32", name="a0")
        a1 = gb.tensor([10240, 16], "float32", name="a1")
        b = gb.emit("Abs", a0, 'b')
        c = gb.emit("Abs", a1, 'c')
        d = gb.emit("Abs", b, 'd')
        e = gb.emit('Add', [c, d], 'e')
        gb.emit("Abs", e, 'f')
    return gb.get()[0]


def graph_4():
    ''' no ring, 2 sibling nodes in 1 step '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([10240, 16], "float32", name="a0")
        a1 = gb.tensor([10240, 16], "float32", name="a1")
        b = gb.emit("Abs", a0, 'b')
        c = gb.emit("Abs", b, 'c')
        d = gb.emit("Abs", a1, 'd')
        e = gb.emit("Abs", d, 'e')
        f = gb.emit('Add', [c, e], 'f')
        gb.emit('Abs', f, 'g')
        h = gb.emit("Abs", d, 'h')
        i = gb.emit('Add', [c, h], 'i')
        gb.emit("Abs", i, 'j')
    return gb.get()[0]


def graph_5():
    ''' no ring, 2 sibling step '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main") as g:
        a0 = gb.tensor([10240, 16], "float32", name="a0")
        a1 = gb.tensor([10240, 16], "float32", name="a1")
        a2 = gb.tensor([10240, 16], "float32", name="a2")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a1, 'b')
        c = gb.emit("Abs", b, 'c')
        d = gb.emit('Add', [a, c], 'd')
        gb.emit("Abs", d, 'e')
        f = gb.emit("Abs", a2, 'f')
        g = gb.emit('Add', [c, f], 'g')
        gb.emit("Abs", g, 'h')
    return gb.get()[0]


def graph_6():
    ''' no ring, tree down '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([10240, 16], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        gb.emit("Abs", b, 'd')
        gb.emit("Abs", b, 'e')
        c = gb.emit("Abs", a, 'c')
        gb.emit("Abs", c, 'f')
        gb.emit("Abs", c, 'g')
    return gb.get()[0]


def graph_pat_1():
    ''' split by reduce '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("ReduceSum", b, 'c', attrs={'reduce_axis': (1,)})
        d = gb.emit("Sqrt", c, 'd')
        gb.emit("Sqrt", d, 'f')
    return gb.get()[0]


def graph_pat_2():
    ''' multi output '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        gb.emit("ReduceSum", b, 'c', attrs={'reduce_axis': (1,)})
        gb.emit("ReduceSum", b, 'e', attrs={'reduce_axis': (1,)})
    return gb.get()[0]


def graph_pat_3():
    ''' two reduce '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("ReduceSum", b, 'c', attrs={'reduce_axis': (1,)})
        d = gb.emit("Abs", c, 'd')
        gb.emit("ReduceSum", d, 'e', attrs={'reduce_axis': (1,)})
    return gb.get()[0]


def graph_pat_4():
    ''' elewise + broadcast '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([1, 1024], "float32", name="a0")
        a2 = gb.tensor([1014, 1024], "float32", name="a2")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("Abs", b, 'c')
        d = gb.emit("Abs", c, 'd')
        e = gb.emit("Abs", d, 'e')
        f = gb.emit("Abs", e, 'f')
        g0 = gb.emit("Abs", a2, 'g0')
        # g0 = gb.emit("Abs", g0, 'g0')
        # g0 = gb.emit("Abs", g0, 'g0')
        # g0 = gb.emit("Abs", g0, 'g0')
        # g0 = gb.emit("Abs", g0, 'g0')
        # g0 = gb.emit("Abs", g0, 'g0')
        # g0 = gb.emit("Abs", g0, 'g0')
        g0 = gb.emit("Abs", g0, 'g0')
        g1 = gb.emit('Add', [f, g0], 'g1')
        g2 = gb.emit("Abs", g1, 'g2')
        g3 = gb.emit("Abs", g2, 'g3')
        g4 = gb.emit("Abs", g3, 'g4')
        gb.emit("Abs", g4, 'g5')
    return gb.get()[0]


def graph_pat_5():
    ''' reduce + reshape '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("ReduceSum", b, 'c', attrs={'reduce_axis': (1,)})
        d = gb.emit("Abs", c, 'd')
        e = gb.tensor([512, 2048], "float32", name="e")
        gb.op("Reshape", e, [d])
    return gb.get()[0]


def graph_pat_6():
    ''' dimond '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("Abs", a, 'c')
        gb.emit("Add", [b, c], 'd')
        gb.emit("Abs", c, 'f')  # broke dimond
    return gb.get()[0]


def graph_pat_7():
    ''' buddy of control op '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a1 = gb.tensor([1024, 1024], "float32", name="a1")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a1, 'b')
        c = gb.emit("MakeTuple", [a, b], 'c')
        d = gb.tensor([1024, 1024], "float32", name="d")
        gb.op("AddN", d, [c])
        gb.emit("Abs", d, 'f')
    graph = gb.get()[0]
    estimate.AddControlBuddy().visit_graph(graph)
    return graph


def graph_pat_8():
    ''' reduce + reshape '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        #c = gb.emit("Abs", b, 'b')
        c = gb.emit("ReduceSum", b, 'c', attrs={'reduce_axis': (1,)})
        gb.emit("Add", [b, c], 'd')
    return gb.get()[0]


def graph_pat_9():
    ''' scalar  '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a1 = gb.tensor([1], "float32", name="a1")
        a = gb.emit("Maximum", a1, 'a')
        b = gb.emit("Mul", [a, a1], 'b')
        gb.emit('Mul', [b, a0], 'c')
    return gb.get()[0]


def graph_mo_1():
    gb = model.GraphBuilder()
    with gb.graph_scope("main"):
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        gb.emit("Abs", a, 'b')
        gb.emit("Abs", a, 'c')
    return gb.get()[0]


def graph_mo_2():
    gb = model.GraphBuilder()
    with gb.graph_scope("main") as g:
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("Abs", b, 'c')
        g.set_output(b, c)
    return gb.get()[0]


def graph_mo_3():
    ''' two reduce '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main") as g:
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("ReduceSum", b, 'c', attrs={'reduce_axis': (1,)})
        g.set_output(b, c)
    return gb.get()[0]


def graph_mo_4():
    ''' two reduce '''
    gb = model.GraphBuilder()
    with gb.graph_scope("main") as g:
        a0 = gb.tensor([1024, 1024], "float32", name="a0")
        a = gb.emit("Abs", a0, 'a')
        b = gb.emit("Abs", a, 'b')
        c = gb.emit("ReduceSum", a, 'c', attrs={'reduce_axis': (1,)})
        g.set_output(b, c)
    return gb.get()[0]


def test_binary_split():
    """Test binary split"""
    def _test(graph, expected_space_size):
        print("********* test on graph : {} *************".format(graph.name))
        sp = split.GraphSpliter(graph)
        nodes = get_nodes(sp, graph.ops)
        space = sp.binary_split(nodes)
        for i, s in enumerate(space):
            print('{}: {}'.format(i, split_format(sp, s)))
        assert len(space) == expected_space_size
        assert first_connected(sp, space)
    _test(graph_1(), 3)
    _test(graph_2(), 7)
    _test(graph_3(), 4)
    _test(graph_4(), 17)
    _test(graph_5(), 11)
    _test(graph_6(), 24)


def test_resolve_connnected_graphs():
    """Test resolve connected graphs"""
    graph = graph_5()
    sp = split.GraphSpliter(graph)
    n1 = get_nodes(sp, ['a', 'd', 'b', 'c'])
    graphs = sp.resolve_connnected_graphs(n1)
    print(graphs)
    assert len(graphs) == 1
    n2 = get_nodes(sp, ['a', 'd', 'e', 'f', 'g'])
    graphs = sp.resolve_connnected_graphs(n2)
    print(graphs)
    assert len(graphs) == 2
    n3 = get_nodes(sp, ['a', 'b', 'f'])
    graphs = sp.resolve_connnected_graphs(n3)
    print(graphs)
    assert len(graphs) == 3


def test_split():
    """Test split"""
    def _print_cost(name, c):
        print("%s\tdma_ratio=%f, saturation=%f, mix_saturation=%f, type=%s" %
              (name, c.dma_ratio(), c.saturation(), c.mix_saturation(), c.cost_type()))

    def _test(graph):
        print("********* test on graph : {} *************".format(graph.name))
        sp = split.GraphSpliter(graph)
        subgraphs = sp.split(False)
        print('----- main graph -------')
        print(graph)
        for i, g in enumerate(subgraphs):
            print(' -------- subgraph {} -------'.format(i))
            print(g)
        print("--------- cost ------------")
        cost, _ = model.estimate(graph)
        _print_cost("main graph", cost)
        fc, sub_costs = model.estimate(subgraphs)
        _print_cost("Subgraphs:", fc)
        for i, cost in enumerate(sub_costs):
            _print_cost(" |_%d:\t" % (i), cost)
    _test(graph_5())
    # _test(graph_4())


def test_estimate():
    """Test estimate"""
    graph = graph_5()
    e = estimate.Estimator(graph)
    e.estimate()
    print(e.iter_space)


def test_pattern_split():
    """Test pattern split"""
    def _test(graph, expect_n=0):
        print("************* main graph **************")
        print(graph)
        subgraphs = split.GraphSplitByPatternV2(graph).split()
        for i, g in enumerate(subgraphs):
            print(' -------- subgraph {} -------'.format(i))
            print(g)
        if expect_n > 0:
            assert len(subgraphs) == expect_n

    # _test(graph_1(), 1)
    # _test(graph_pat_1(), 2)
    # _test(graph_pat_2())
    # _test(graph_pat_3())
    # _test(graph_pat_4())
    # _test(graph_pat_5())
    # _test(graph_pat_6())
    # _test(graph_pat_7())
    # _test(graph_pat_8())
    # _test(graph_pat_9())

    # _test(graph_mo_1())
    # _test(graph_mo_2())
    # _test(graph_mo_3())
    _test(graph_mo_4())


def main():
    # test_binary_split()
    # test_resolve_connnected_graphs()
    # test_split()
    # test_estimate()
    test_pattern_split()


if __name__ == '__main__':
    main()
