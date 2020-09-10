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
"""Cost model splitter"""

from .model import PrimLib, Graph


class GraphSplitByPattern:
    """Graph split by pattern"""

    def __init__(self, graph):
        self.graph = graph
        self.groups = []
        self.op_group = {}
        for op in self.graph.ops:
            g = [op]
            self.groups.append(g)
            self.op_group[op] = g
        self.ids = {}
        for i, op in enumerate(graph.ops):
            self.ids[op] = i
        self.doms = self.post_dom(graph.ops)
        _, outputs = graph.deduce_parameters()
        self.outputs = set(outputs)

    def post_dom(self, ops):
        """Post dom"""
        doms, i_doms = {}, {}
        for i in range(len(ops) - 1, -1, -1):
            op = ops[i]
            doms[op] = {op}
            i_dom = None
            if op.output.to_ops:
                suc_dom = set(doms[op.output.to_ops[0]])
                for to in op.output.to_ops[1:]:
                    suc_dom.intersection_update(doms[to])
                doms[op].update(suc_dom)
                for dom in suc_dom:
                    if i_dom is None or self.ids[dom] < self.ids[i_dom]:
                        i_dom = dom
            i_doms[op] = i_dom
        return i_doms

    def get_pattern(self, op, i):
        """Get pattern"""
        pattern = PrimLib.UNKNOWN
        _, elem_relation = PrimLib.input_relation(op, i)
        for pat in elem_relation:
            if pat and pat > pattern:
                pattern = pat
        return pattern

    def fuse(self, check_fun):
        """Fuse ops"""
        def _get_path(op, dom):
            path_ops, visited = [], set()

            def _get_path_depth(p):
                visited.add(p)
                if self.op_group[p][0] == p:
                    path_ops.append(p)
                for to in p.output.to_ops:
                    if to != dom and to not in visited:
                        _get_path_depth(to)
            _get_path_depth(op)
            return path_ops
        changed = True
        while changed:
            for group in self.groups:
                op = group[0]
                dom = self.doms[op]
                if dom is None or op.output in self.outputs:
                    continue
                ops = _get_path(op, dom)
                if check_fun(op, dom, ops):
                    dom_group = self.op_group[dom]
                    fused = []
                    for fop in ops:
                        f_group = self.op_group[fop]
                        for p in f_group:
                            self.op_group[p] = dom_group
                        fused.append(f_group)
                        dom_group += f_group
                    for g in fused:
                        self.groups.remove(g)
                    break
            else:
                changed = False

    def to_subgraphs(self):
        """Transform op groups to subgraphs"""
        subgraphs = []
        for i, group in enumerate(self.groups):
            group.sort(key=lambda op: self.ids[op])
            subgraphs.append(Graph('{}_{}'.format(self.graph.name, i), group))
        return subgraphs

    def split(self):
        """Split graph"""
        def _buddy(op, dom, path_ops):
            """Fuse buddy together"""
            # pylint: disable=unused-argument
            group = self.op_group[op]
            for p in group:
                # p is buddy
                if p.output.buddy is not None and p.output.buddy.members[0].op not in group:
                    return True
                # p's output is buddy
                for to in p.output.to_ops:
                    if to.output.buddy is not None and to not in group:
                        return True
            return False

        def _injective(pattern, limit):
            def _checker(op, dom, path_ops):
                # pylint: disable=unused-argument
                for p in op.output.to_ops:
                    if p not in self.op_group[dom]:
                        return False
                if PrimLib.iter_type(op) in (PrimLib.ELEMWISE, PrimLib.BROADCAST):
                    for i, t in enumerate(dom.inputs):
                        if t == op.output:
                            return self.get_pattern(dom, i) == pattern and len(self.op_group[op]) < limit
                return False
            return _checker

        def _diamond(op, dom, path_ops):
            if PrimLib.iter_type(op) not in (PrimLib.ELEMWISE, PrimLib.BROADCAST) or \
                    PrimLib.iter_type(dom) in (PrimLib.UNKNOWN, PrimLib.TRANSFORM):
                return False
            return len(path_ops) == 1 and op.output not in dom.inputs
        self.fuse(_buddy)
        self.fuse(_injective(PrimLib.ELEMWISE, 100))
        self.fuse(_injective(PrimLib.BROADCAST, 6))
        self.fuse(_injective(PrimLib.REDUCE, 6))
        self.fuse(_diamond)
        return self.to_subgraphs()


def split(graph):
    return GraphSplitByPattern(graph).split()
