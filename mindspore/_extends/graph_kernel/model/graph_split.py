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

from .model import PrimLib, Graph, Tensor


class GraphSplitByPattern:
    """Graph splitter"""
    class Area:
        """Area"""
        MODE_BASIC = 1
        MODE_COMPOSITE = 2

        def __init__(self, init_op):
            self.pattern = PrimLib.iter_type(init_op)
            self.ops = [init_op]
            self.in_relations = dict()  # {area1: relation1, area2: relation2, ...}
            self.out_relations = dict()  # {area1: relation1, area2: relation2, ...}
            self.mode = self.MODE_BASIC
            if self.pattern == PrimLib.TRANSFORM:
                self.mode = self.MODE_COMPOSITE

        def __str__(self):
            return '<' + '-'.join([op.output.name for op in self.ops]) + '>'

        def __repr__(self):
            return str(self)

        def link_input(self, area_map):
            """Link inputs"""
            def get_relation(op, i):
                relation = PrimLib.UNKNOWN
                _, elem_relation = PrimLib.input_relation(op, i)
                for r in elem_relation:
                    if r is not None and r > relation:
                        relation = r
                return relation
            for i, t in enumerate(self.ops[0].inputs):
                if t.op is not None:
                    area, relation = area_map[t.op], get_relation(self.ops[0], i)
                    self.in_relations[area] = relation

        def link_output(self):
            """Link outputs"""
            for input_area, r in self.in_relations.items():
                input_area.out_relations[self] = r

        def fuse(self, area):
            """Fuse `area` to `self`"""
            def _update_relation(relations, a, r):
                relations[a] = max(r, relations[a]) if a in relations else r

            def _update_pattern():
                self.pattern = max(self.pattern, area.pattern, self.in_relations[area])

            def _fuse_relation(self_relations, new_relations):
                for a, r in new_relations.items():
                    if a != self:
                        _update_relation(self_relations, a, r)
                if area in self_relations:
                    self_relations.pop(area)

            def _redirect_relation(rels):
                """Replace `area` with `self` in relations"""
                if area in rels:
                    r = rels.pop(area)
                    _update_relation(rels, self, r)

            self.ops.extend(area.ops)
            _update_pattern()
            _fuse_relation(self.in_relations, area.in_relations)
            _fuse_relation(self.out_relations, area.out_relations)
            for a, _ in area.in_relations.items():
                _redirect_relation(a.out_relations)
            for a, _ in area.out_relations.items():
                _redirect_relation(a.in_relations)
            self.mode = self.MODE_COMPOSITE

        def check_circle(self, to):
            """Check circle. It returns false if circle exists"""
            def _reached(area, to):
                for out, _ in area.out_relations.items():
                    if out == to or _reached(out, to):
                        return True
                return False
            for out, _ in self.out_relations.items():
                if out != to and _reached(out, to):
                    return False
            return True

    BORADCAST_FUSE_DEPTH = 3
    REDUCE_FUSE_DEPTH = 3

    def __init__(self, graph):
        self.graph = graph
        self.areas = []
        area_map = {}
        for op in graph.ops:
            a = self.Area(op)
            self.areas.append(a)
            area_map[op] = a
        for a in self.areas:
            a.link_input(area_map)
        for a in self.areas:
            a.link_output()

    def fuse(self, selector):
        """Fuse areas"""
        changed = False
        while True:
            for dominant in self.areas:
                fuse_areas = selector(dominant)
                if fuse_areas:
                    for area in fuse_areas:
                        changed = True
                        dominant.fuse(area)
                        self.areas.remove(area)
                    break
            else:
                return changed

    def to_subgraphs(self):
        """Transform op groups to subgraphs"""
        ids = {}
        for i, op in enumerate(self.graph.ops):
            ids[op] = i
        subgraphs = []
        graphmodes = []
        for i, area in enumerate(self.areas):
            area.ops.sort(key=lambda op: ids[op])
            subgraphs.append(Graph('{}_{}'.format(self.graph.name, i), area.ops))
            graphmodes.append("basic" if area.mode == self.Area.MODE_BASIC else "composite")
        return subgraphs, graphmodes

    def split(self):
        """Split graph by pattern"""
        def _elemwise_depth(dom):
            if dom.pattern > PrimLib.BROADCAST or len(dom.in_relations) != 1:
                return None
            a, r = list(dom.in_relations.items())[0]
            if a.pattern > PrimLib.BROADCAST or len(a.out_relations) != 1 and r != PrimLib.ELEMWISE:
                return None
            return [a]

        def _elemwise_width(dom):
            if dom.pattern > PrimLib.BROADCAST:
                return None
            fused = []
            for a, r in dom.in_relations.items():
                if a.pattern <= PrimLib.BROADCAST and r == PrimLib.ELEMWISE and a.check_circle(dom):
                    fused.append(a)
            return fused

        def _broadcast_depth(dom):
            if dom.pattern > PrimLib.BROADCAST or len(dom.in_relations) != 1:
                return None
            a, r = list(dom.in_relations.items())[0]
            if a.pattern > PrimLib.BROADCAST or len(a.out_relations) != 1 or \
                    r != PrimLib.BROADCAST or len(a.ops) > self.BORADCAST_FUSE_DEPTH:
                return None
            return [a]

        def _broadcast_width(dom):
            if dom.pattern > PrimLib.BROADCAST:
                return None
            fused = []
            for a, r in dom.in_relations.items():
                if a.pattern <= PrimLib.BROADCAST and r == PrimLib.BROADCAST and \
                        a.check_circle(dom) and len(a.ops) <= self.BORADCAST_FUSE_DEPTH:
                    fused.append(a)
            return fused

        def _check_reduce_exclude(dom):
            # exclude large all-reduce
            if len(dom.ops[0].inputs[0].shape) == len(dom.ops[0].attrs["reduce_axis"]) and \
                    dom.ops[0].inputs[0].get_size() > 10000:
                return True

            # exclude multi output
            for a in dom.in_relations.keys():
                if len(a.out_relations) > 1:
                    return True
                if any([op.output.para_type == Tensor.PARA_OUTPUT for op in a.ops]):
                    return True
            return False

        def _reduce_depth(dom):
            if dom.pattern != PrimLib.REDUCE or len(dom.in_relations) != 1:
                return None
            if _check_reduce_exclude(dom):
                return None
            a, r = list(dom.in_relations.items())[0]
            if a.pattern > PrimLib.BROADCAST or len(a.out_relations) != 1 or \
                    r > PrimLib.REDUCE or len(a.ops) > self.REDUCE_FUSE_DEPTH:
                return None
            return [a]

        def _reduce_width(dom):
            if dom.pattern != PrimLib.REDUCE:
                return None
            if _check_reduce_exclude(dom):
                return None
            fused = []
            for a, r in dom.in_relations.items():
                if a.pattern <= PrimLib.BROADCAST and r <= PrimLib.REDUCE and \
                        a.check_circle(dom) and len(a.ops) <= self.REDUCE_FUSE_DEPTH:
                    fused.append(a)
            return fused
        changed = True
        while changed:
            changed = self.fuse(_elemwise_depth)
            changed = self.fuse(_elemwise_width) or changed
            changed = self.fuse(_broadcast_depth) or changed
            changed = self.fuse(_broadcast_width) or changed
            changed = self.fuse(_reduce_depth) or changed
            changed = self.fuse(_reduce_width) or changed
        subgraphs, graphmodes = self.to_subgraphs()
        return subgraphs, graphmodes


def split(graph):
    """Split graph"""
    return GraphSplitByPattern(graph).split()
