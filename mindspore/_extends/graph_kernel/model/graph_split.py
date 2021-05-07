# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import os
from functools import reduce
from mindspore import log as logger
from .model import PrimLib, Graph, Tensor
from .model import DataFormat as DF

class GraphSplitByPattern:
    """Graph splitter"""
    class ReachTable:
        """Reachable table"""
        def __init__(self, size):
            self.map = []
            self.alive = set(range(size))
            for i in range(0, size):
                self.map.append([False for j in range(0, size)])
                self.map[i][i] = True

        def reachable(self, x, y):
            """reachable from x to y"""
            return self.map[x][y]

        def sync(self, x, y):
            """sync from y to x"""
            for i in self.alive:
                if self.map[y][i] and not self.map[x][i]:
                    self.map[x][i] = True

        def fuse(self, x, y):
            """fuse y to x"""
            for i in self.alive:
                if self.map[y][i] and not self.map[x][i]:
                    self.map[x][i] = True
                if self.map[i][y] and not self.map[i][x]:
                    self.map[i][x] = True
            self.alive.remove(y)

    class Area:
        """Area"""
        MODE_BASIC = 1
        MODE_COMPOSITE = 2

        class StitchInfo:
            """StitchInfo"""
            def __init__(self):
                self.stitch_ops = set()
                self.stitch_atomic_ops = set()

        def __init__(self, init_op, is_output, unique_id, reach_tab):
            self.pattern = PrimLib.iter_type(init_op)
            self.ops = [init_op]
            self.in_relations = dict()  # {area1: relation1, area2: relation2, ...}
            self.out_relations = dict()  # {area1: relation1, area2: relation2, ...}
            self.mode = None
            self.stitch_info = self.StitchInfo()
            self.is_output = is_output
            self.output_excluded = set()
            if self.pattern == PrimLib.REDUCE:
                def _gather_reduce_exclude(op):
                    for to in op.output.to_ops:
                        idx = to.inputs.index(op.output)
                        if self.get_relation(to, idx) > PrimLib.ELEMWISE:
                            self.output_excluded.add(to)
                        else:
                            _gather_reduce_exclude(to)
                _gather_reduce_exclude(init_op)
            self.unique_id = unique_id
            self.reach_tab = reach_tab

        def __str__(self):
            return '<' + '-'.join([op.output.name for op in self.ops]) + '>'

        def __repr__(self):
            return str(self)

        def get_relation(self, op, i):
            relation = PrimLib.UNKNOWN
            _, elem_relation = PrimLib.input_relation(op, i)
            for r in elem_relation:
                if r is None:
                    relation = max(relation, PrimLib.BROADCAST)
                elif r > relation:
                    relation = r
            return relation

        def link_input(self, area_map):
            """Link inputs"""
            for i, t in enumerate(self.ops[0].inputs):
                if t.op is not None:
                    area, relation = area_map[t.op], self.get_relation(self.ops[0], i)
                    self.in_relations[area] = relation

        def link_output(self):
            """Link outputs"""
            for input_area, r in self.in_relations.items():
                input_area.out_relations[self] = r
            for out, _ in self.out_relations.items():
                self.reach_tab.sync(self.unique_id, out.unique_id)

        def update_stitch_info(self, stitch_info):
            if stitch_info.stitch_ops:
                self.stitch_info.stitch_ops.update(stitch_info.stitch_ops)
            if stitch_info.stitch_atomic_ops:
                self.stitch_info.stitch_atomic_ops.update(stitch_info.stitch_atomic_ops)

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

            if self.pattern >= area.pattern:
                self.ops.extend(area.ops)
            else:
                self.ops = area.ops + self.ops
            _update_pattern()
            _fuse_relation(self.in_relations, area.in_relations)
            _fuse_relation(self.out_relations, area.out_relations)
            for a, _ in area.in_relations.items():
                _redirect_relation(a.out_relations)
            for a, _ in area.out_relations.items():
                _redirect_relation(a.in_relations)
            if self.pattern > PrimLib.RESHAPE:
                self.mode = self.MODE_COMPOSITE
            if area.is_output and not self.is_output:
                self.is_output = True
            if area.output_excluded:
                self.output_excluded.update(area.output_excluded)
            self.update_stitch_info(area.stitch_info)
            self.reach_tab.fuse(self.unique_id, area.unique_id)

        def check_acyclic(self, to):
            """Check circle. It returns false if circle exists"""
            for out, _ in self.out_relations.items():
                if out != to and self.reach_tab.reachable(out.unique_id, to.unique_id):
                    return False
            return True

        def dom_op(self):
            return self.ops[0]

        def reduce_out_exclude(self, area):
            if self.output_excluded:
                for op in self.output_excluded:
                    if op in area.ops:
                        return True
            return False

    def __init__(self, graph, flags):
        self.graph = graph
        self.areas = []
        self.flags = flags
        self.reach_tab = self.ReachTable(len(graph.ops))
        area_map = {}
        _, outputs = graph.deduce_parameters()
        idx = 0
        for op in graph.ops:
            is_output = op.output in outputs
            a = self.Area(op, is_output, idx, self.reach_tab)
            idx += 1
            self.set_default_mode(a)
            self.areas.append(a)
            area_map[op] = a
        for a in self.areas:
            a.link_input(area_map)
        for i in range(len(self.areas)-1, -1, -1):
            self.areas[i].link_output()

    def set_default_mode(self, area):
        area.mode = self.get_default_mode(area.ops[0])

    def fuse(self, selector):
        """Fuse areas"""
        changed = False
        while True:
            for dominant in self.areas:
                result = selector(dominant)
                if result is not None and result[0]:
                    fuse_areas, is_forward = result
                    if is_forward:
                        for area in fuse_areas:
                            dominant.fuse(area)
                            self.areas.remove(area)
                    else:
                        forward_area = dominant
                        for area in fuse_areas:
                            area.fuse(forward_area)
                            self.areas.remove(forward_area)
                            forward_area = area
                    changed = True
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
            subgraphs.append(Graph('{}_{}'.format(self.graph.name, i), area.ops, area.stitch_info))
            graphmodes.append("basic" if area.mode == self.Area.MODE_BASIC else "composite")
        return subgraphs, graphmodes

    def dump_subgraphs(self, subgraphs):
        """Dump subgraphs"""
        if os.environ.get("ENABLE_SUBGRAPHS", "off") == "on":
            subgraphs_str = "subgraphs:\nlen: " + str(len(subgraphs)) + "\n"
            for i, sub in enumerate(subgraphs):
                subgraphs_str += str("============") + str(i) + "\n"
                subgraphs_str += str(sub)
            dirname = 'subgraphs'
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            graphname = self.graph.name
            filename = dirname + '/' + graphname + '.log'
            with open(filename, 'w') as f:
                f.write(subgraphs_str)

    def do_split(self):
        """Split graph by pattern"""
        raise Exception("do_split() is not implemented in {}".format(self.__class__.__name__))

    def split(self):
        """Split graph by pattern"""
        self.do_split()
        # The reshape should not be output node
        # Note: after this function, the input output relation is not maintained.
        self.split_output_reshapes()
        subgraphs, graphmodes = self.to_subgraphs()
        self.dump_subgraphs(subgraphs)
        return subgraphs, graphmodes

    def split_output_reshapes(self):
        """Force split the output reshapes into other new """
        new_areas = []
        for area in self.areas:
            out_reshape_ops = [op for op in area.ops if PrimLib.iter_type(op) == PrimLib.RESHAPE]
            remain_ops = [op for op in area.ops if op not in out_reshape_ops]
            if not remain_ops or not out_reshape_ops:
                continue
            changed = True
            while changed:
                changed = False
                for op in out_reshape_ops:
                    if any([to_op in remain_ops for to_op in op.output.to_ops]):
                        out_reshape_ops.remove(op)
                        remain_ops.append(op)
                        changed = True
                        break
            if out_reshape_ops:
                for op in out_reshape_ops:
                    a = self.Area(op, False, 0, self.reach_tab)
                    self.set_default_mode(a)
                    new_areas.append(a)
                area.ops = remain_ops
                if len(remain_ops) == 1:
                    self.set_default_mode(area)
        if new_areas:
            self.areas += new_areas

use_poly_reduce = True
class GraphSplitGpu(GraphSplitByPattern):
    """Graph splitter"""
    BORADCAST_FUSE_DEPTH = 20
    REDUCE_FUSE_DEPTH = 20

    def get_default_mode(self, op):
        pattern = PrimLib.iter_type(op)
        return self.Area.MODE_BASIC if pattern == PrimLib.RESHAPE else self.Area.MODE_COMPOSITE

    def do_split(self):
        """Split graph by pattern"""
        def _reshape(dom):
            if dom.pattern != PrimLib.RESHAPE:
                return None
            min_area, forward_fuse = None, False
            for a, _ in dom.out_relations.items():
                if a.pattern <= PrimLib.BROADCAST and dom.check_acyclic(a) and \
                        (min_area is None or a.pattern < min_area.pattern):
                    min_area = a
            for a, _ in dom.in_relations.items():
                if a.pattern <= PrimLib.BROADCAST and a.check_acyclic(dom) and \
                   len(dom.ops[0].inputs[0].to_ops) == 1 and not a.is_output and \
                   (min_area is None or a.pattern < min_area.pattern):
                    min_area, forward_fuse = a, True
            return ([min_area], forward_fuse) if min_area else None

        def _elemwise_depth(dom):
            if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST) or len(dom.in_relations) != 1:
                return None
            a, r = list(dom.in_relations.items())[0]
            if a.pattern > PrimLib.BROADCAST or len(a.out_relations) != 1 or r != PrimLib.ELEMWISE or \
                    a.dom_op().output.shape != dom.dom_op().output.shape:
                return None
            return [a], True

        def _elemwise_width(dom):
            if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST):
                return None
            fused = []
            for a, r in dom.in_relations.items():
                if a.pattern <= PrimLib.BROADCAST and r == PrimLib.ELEMWISE and a.check_acyclic(dom) and \
                        a.dom_op().output.shape == dom.dom_op().output.shape:
                    fused.append(a)
            return fused, True

        def _broadcast_pat_exclude(dom, a, r):
            if use_poly_reduce and a.pattern == PrimLib.REDUCE:
                return dom.pattern > PrimLib.ELEMWISE or r > PrimLib.ELEMWISE
            return a.pattern > PrimLib.REDUCE or r > PrimLib.BROADCAST

        def _broadcast_depth(dom):
            if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST) or len(dom.out_relations) != 1 or \
                    dom.is_output or len(dom.ops) > self.BORADCAST_FUSE_DEPTH:
                return None
            a, r = list(dom.out_relations.items())[0]
            if _broadcast_pat_exclude(dom, a, r) or len(a.in_relations) != 1:
                return None
            return [a], False

        def _broadcast_width(dom):
            if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST) or \
                    dom.is_output or len(dom.ops) > self.BORADCAST_FUSE_DEPTH:
                return None
            fused = []
            for a, r in dom.out_relations.items():
                if _broadcast_pat_exclude(dom, a, r) or not dom.check_acyclic(a) or \
                        (fused and fused[0].dom_op().output.shape != a.dom_op().output.shape):
                    return None
                fused.append(a)
            return fused, False

        def _check_reduce_exclude(dom):
            if use_poly_reduce:
                return False
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

        def _reduce_pat_exclude(_, a, r):
            if len(a.ops) > self.REDUCE_FUSE_DEPTH:
                return True
            if use_poly_reduce:
                return a.pattern > PrimLib.ELEMWISE or r > PrimLib.REDUCE or r == PrimLib.BROADCAST
            return a.pattern > PrimLib.BROADCAST or r > PrimLib.REDUCE

        def _reduce_depth(dom):
            if dom.pattern != PrimLib.REDUCE or len(dom.in_relations) != 1:
                return None
            if _check_reduce_exclude(dom):
                return None
            a, r = list(dom.in_relations.items())[0]
            if dom.ops[0].inputs[0].dtype == "float16" and a.is_output and len(a.ops) >= 10 and \
                    _is_atomic_add_available(dom):
                # to evade the precision problem.
                return None
            if _reduce_pat_exclude(dom, a, r) or len(a.out_relations) != 1:
                return None
            return [a], True

        def _reduce_width(dom):
            if dom.pattern != PrimLib.REDUCE:
                return None
            if _check_reduce_exclude(dom):
                return None
            fused = []
            for a, r in dom.in_relations.items():
                if dom.ops[0].inputs[0].dtype == "float16" and a.is_output and len(a.ops) >= 10 and \
                        _is_atomic_add_available(dom):
                    # to evade the precision problem.
                    continue
                if not _reduce_pat_exclude(dom, a, r) and a.check_acyclic(dom):
                    fused.append(a)
            return fused, True

        def _tensor_size(tensor):
            size = 1
            for i in tensor.shape:
                size *= i
            return size

        def _is_atomic_add_available(dom):
            if any(["Reduce" in x.prim for x in dom.ops[1:]]):
                return False
            op = dom.ops[0]
            reduce_axis = op.attrs["reduce_axis"]
            if len(op.inputs[0].shape) - 1 in reduce_axis:
                reduce_size = reduce(lambda x, y: x * y, [op.inputs[0].shape[i] for i in reduce_axis])
                return reduce_size >= 1024
            return True

        def _reduce_nums(ops):
            count = 0
            for op in ops:
                if op.prim.startswith('Reduce'):
                    count += 1
            return count

        def _reduce_output(dom):
            if dom.pattern != PrimLib.REDUCE:
                return None
            if _reduce_nums(dom.ops) > 1:
                return None
            if _is_atomic_add_available(dom):
                return None
            is_all_reduce = _tensor_size(dom.ops[0].output) == 1
            # excluded large size all reduce
            if is_all_reduce and _tensor_size(dom.ops[0].inputs[0]) > 1024 * 12:
                return None

            fused = []
            for a, r in dom.out_relations.items():
                if a.pattern <= PrimLib.BROADCAST and r <= PrimLib.BROADCAST and \
                        dom.check_acyclic(a) and not dom.reduce_out_exclude(a):
                    fused.append(a)
            return fused, False

        def _stitch_axis(shape):
            stitch_axis = []
            size = 1
            for i in shape:
                size = size * i
                stitch_axis.append(i)
                if size >= 1024 * 8:
                    return stitch_axis
            return None

        def _same_stitch_axis(a, b):
            x = []
            x.extend(a)
            x.extend(b)
            stitch_axis = _stitch_axis(x[0].shape)
            for item in x:
                i_stitch_axis = _stitch_axis(item.shape)
                if i_stitch_axis is None or i_stitch_axis != stitch_axis:
                    return False
            return True

        def _may_stitch(dom, a, r):
            if a.pattern <= PrimLib.REDUCE and r <= PrimLib.BROADCAST and dom.check_acyclic(a):
                if _reduce_nums(a.ops) < 2:
                    dom_outs = [op.output for op in dom.ops]
                    a_ins = [input for op in a.ops for input in op.inputs]
                    a_outs = [op.output for op in a.ops]
                    a_final_outs = [tensor for tensor in a_outs if tensor not in a_ins]
                    stitch_tensors = [tensor for tensor in dom_outs if tensor in a_ins]
                    if _same_stitch_axis(stitch_tensors, a_final_outs):
                        for tensor in stitch_tensors:
                            if _tensor_size(tensor) >= 1024 * 1024 * 12:
                                return True
            return False

        def _reduce_stitch(dom):
            if dom.pattern != PrimLib.REDUCE:
                return None
            if _tensor_size(dom.ops[0].output) == 1:
                return None
            if _tensor_size(dom.ops[0].inputs[0]) < 1024 * 12:
                return None

            fused = []
            for a, r in dom.out_relations.items():
                if _may_stitch(dom, a, r):
                    if a.pattern == PrimLib.REDUCE:
                        if a.ops[0].attrs['reduce_axis'] == dom.ops[0].attrs['reduce_axis']:
                            dom.stitch_info.stitch_ops.add(dom.ops[0].output.name)
                            fused.append(a)
                    elif a.pattern == PrimLib.BROADCAST:
                        dom.stitch_info.stitch_ops.add(dom.ops[0].output.name)
                        fused.append(a)
            return fused, False

        def _transpose(dom):
            if len(dom.ops) != 1 or dom.ops[0].prim != "Transpose":
                return None
            fused = []
            for a, _ in dom.in_relations.items():
                if a.pattern <= PrimLib.BROADCAST and a.check_acyclic(dom):
                    fused.append(a)
            return fused, True

        enable_stitch_fusion = self.flags.get("enable_stitch_fusion", False)
        changed = True
        while changed:
            changed = self.fuse(_reshape)
            changed = self.fuse(_elemwise_depth) or changed
            changed = self.fuse(_elemwise_width) or changed
            changed = self.fuse(_reduce_depth) or changed
            changed = self.fuse(_reduce_width) or changed
            changed = self.fuse(_broadcast_depth) or changed
            changed = self.fuse(_broadcast_width) or changed
            if use_poly_reduce:
                changed = self.fuse(_reduce_output) or changed
                if enable_stitch_fusion:
                    changed = self.fuse(_reduce_stitch) or changed
        self.fuse(_transpose)

class GraphSplitAscend(GraphSplitByPattern):
    """Graph splitter"""
    BORADCAST_FUSE_DEPTH = 6
    REDUCE_FUSE_DEPTH = 10

    def get_default_mode(self, op):
        if op.prim == "MatMul" or op.prim == "BatchMatMul":
            return self.Area.MODE_COMPOSITE if op.inputs[0].dtype == "float16" else self.Area.MODE_BASIC
        if op.prim in ("Tile", "BroadcastTo", "ExpandDims"):
            return self.Area.MODE_COMPOSITE
        return self.Area.MODE_BASIC

    def do_split(self):
        """Split graph by pattern"""
        def _tensor_size(tensor):
            size = 1
            for i in tensor.shape:
                size *= i
            return size

        def _likely_multicore(dom):
            op = dom.dom_op()
            iter_size = _tensor_size(op.output if not PrimLib.is_reduce(op) else op.inputs[0])
            return iter_size > 1024

        def _reshape(dom):
            if dom.pattern != PrimLib.RESHAPE:
                return None
            min_area, forward_fuse = None, False
            for a, _ in dom.out_relations.items():
                if a.pattern <= PrimLib.BROADCAST and dom.check_acyclic(a) and \
                        (min_area is None or a.pattern < min_area.pattern):
                    min_area = a
            for a, _ in dom.in_relations.items():
                if a.pattern <= PrimLib.BROADCAST and a.check_acyclic(dom) and \
                   len(dom.ops[0].inputs[0].to_ops) == 1 and not a.is_output and \
                   (min_area is None or a.pattern < min_area.pattern):
                    min_area, forward_fuse = a, True
            return ([min_area], forward_fuse) if min_area else None

        def _elemwise_depth(dom):
            if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST) or len(dom.in_relations) != 1:
                return None
            a, r = list(dom.in_relations.items())[0]
            if a.pattern > PrimLib.BROADCAST or len(a.out_relations) != 1 or r != PrimLib.ELEMWISE or \
                    a.dom_op().output.shape != dom.dom_op().output.shape:
                return None
            return [a], True

        def _elemwise_width(dom):
            if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST):
                return None
            fused = []
            for a, r in dom.in_relations.items():
                if a.pattern <= PrimLib.BROADCAST and r == PrimLib.ELEMWISE and a.check_acyclic(dom) and \
                        a.dom_op().output.shape == dom.dom_op().output.shape:
                    fused.append(a)
            return fused, True

        def _broadcast_pat_exclude(dom, a, r):
            if _likely_multicore(a) and (dom.is_output or len(dom.ops) > self.BORADCAST_FUSE_DEPTH):
                return True
            return a.pattern > PrimLib.REDUCE or r > PrimLib.BROADCAST

        def _broadcast_depth(dom):
            if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST) or len(dom.out_relations) != 1:
                return None
            a, r = list(dom.out_relations.items())[0]
            if _broadcast_pat_exclude(dom, a, r) or len(a.in_relations) != 1:
                return None
            return [a], False

        def _broadcast_width(dom):
            if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST):
                return None
            fused = []
            for a, r in dom.out_relations.items():
                if _broadcast_pat_exclude(dom, a, r) or not dom.check_acyclic(a) or \
                        (fused and fused[0].dom_op().output.shape != a.dom_op().output.shape):
                    return None
                fused.append(a)
            return fused, False

        def _reduce_pat_exclude(dom, a, r):
            if len(a.ops) > self.REDUCE_FUSE_DEPTH:
                return True
            if r == PrimLib.BROADCAST and _likely_multicore(dom) and \
                (dom.is_output or len(dom.ops) > self.BORADCAST_FUSE_DEPTH):
                return True
            return a.pattern > PrimLib.BROADCAST or r > PrimLib.REDUCE

        def _reduce_depth(dom):
            if dom.pattern != PrimLib.REDUCE or len(dom.in_relations) != 1:
                return None
            a, r = list(dom.in_relations.items())[0]
            if _reduce_pat_exclude(dom, a, r) or len(a.out_relations) != 1:
                return None
            return [a], True

        def _reduce_width(dom):
            if dom.pattern != PrimLib.REDUCE:
                return None
            fused = []
            for a, r in dom.in_relations.items():
                if not _reduce_pat_exclude(dom, a, r) and a.check_acyclic(dom):
                    fused.append(a)
            return fused, True

        def _matmul_depth(dom):
            if dom.dom_op().prim != "MatMul" and dom.dom_op().prim != "BatchMatMul":
                return None
            fused = []
            for a, _ in dom.out_relations.items():
                if a.pattern == PrimLib.ELEMWISE and a.check_acyclic(dom):
                    fused.append(a)
            return fused, False

        def _transdata_pattern_support(dom, a):
            transdata_op = dom.dom_op()

            # Currently, if transdata has the pad, it is not used to fuse
            def _has_pad():
                res = False
                input_shape = transdata_op.inputs[0].shape
                output_shape = transdata_op.output.shape
                cube_size = 16
                for dim in input_shape[-2:]:
                    if dim % cube_size != 0:
                        res = True
                for dim in output_shape[-2:]:
                    if dim % cube_size != 0:
                        res = True
                return res
            has_pad = _has_pad()
            if has_pad:
                return False

            if a.dom_op().prim == "MatMul" and len(dom.ops) == 1:
                return True

            # reshape/elewise/broadcast + transdata
            if a.pattern <= PrimLib.BROADCAST and len(dom.ops) == 1:
                op_attrs = dom.dom_op().attrs
                if 'src_format' not in op_attrs.keys() \
                        or 'dst_format' not in op_attrs.keys():
                    logger.error("src_format or dst_format not be found in the attrs of Transdata op")
                    return False
                src_format, dst_format = op_attrs['src_format'], op_attrs['dst_format']
                if src_format == DF.FRAC_NZ and dst_format in (DF.DEFAULT, DF.NCHW):
                    return True
                # For the Default/NCHW to FRAC_NZ, currently only the Cast+Transdata is supported
                if src_format in (DF.DEFAULT, DF.NCHW) and dst_format == DF.FRAC_NZ\
                        and len(a.ops) == 1 and a.dom_op().prim == "Cast" and not a.is_output:
                    return True
            return False

        def _transdata(dom):
            if dom.dom_op().prim != "TransData":
                return None
            fused = []
            for a, _ in dom.in_relations.items():
                if _transdata_pattern_support(dom, a) and a.check_acyclic(dom):
                    fused.append(a)
            return fused, True

        def _reduce_output(dom):
            if dom.pattern != PrimLib.REDUCE:
                return None
            op_attrs = dom.dom_op().attrs
            if not op_attrs.get('reduce_output_fuse'):
                return None
            fused = []
            for a, r in dom.out_relations.items():
                if a.pattern <= PrimLib.BROADCAST and r <= PrimLib.BROADCAST and \
                        dom.check_acyclic(a):
                    fused.append(a)
            return fused, False

        changed = True
        while changed:
            changed = self.fuse(_reshape)
            changed = self.fuse(_elemwise_depth) or changed
            changed = self.fuse(_elemwise_width) or changed
            changed = self.fuse(_reduce_depth) or changed
            changed = self.fuse(_reduce_width) or changed
            changed = self.fuse(_broadcast_depth) or changed
            changed = self.fuse(_broadcast_width) or changed
            changed = self.fuse(_matmul_depth) or changed
            changed = self.fuse(_reduce_output) or changed
        self.fuse(_transdata)


def split(graph, target, flags):
    """Split graph"""
    result = None
    if target == "cuda":
        result = GraphSplitGpu(graph, flags).split()
    else:
        result = GraphSplitAscend(graph, flags).split()
    return result
