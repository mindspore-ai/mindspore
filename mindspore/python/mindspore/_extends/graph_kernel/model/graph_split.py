# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from functools import reduce as prod_reduce
from functools import partial
from .model import PrimLib, Graph, Tensor, Operator


def tensor_size(tensor):
    """get tensor size"""
    size = 1
    for i in tensor.shape:
        size *= i
    return size


def reduce_nums(ops):
    """get reduce nums"""
    count = 0
    for op in ops:
        if op.prim.startswith('Reduce'):
            count += 1
    return count


def may_stitch(dom, a, r, stitch_axis_size, stitch_buffer_size):
    """check if can stitch"""

    def _same_stitch_axis(stitch_tensors, final_outs, stitch_axis_size):
        """does a and b have same stitch axis"""

        def _stitch_axis(shape, stitch_axis_size):
            """get stitch axis"""
            stitchaxis = []
            size = 1
            for i in shape:
                size = size * i
                stitchaxis.append(i)
                if size >= stitch_axis_size:
                    return stitchaxis
            return []

        x = []
        x.extend(stitch_tensors)
        x.extend(final_outs)
        stitch_axis_0 = _stitch_axis(x[0].shape, stitch_axis_size)
        for item in x:
            i_stitch_axis = _stitch_axis(item.shape, stitch_axis_size)
            if not i_stitch_axis or i_stitch_axis != stitch_axis_0:
                return False
        return True

    if a.pattern <= PrimLib.REDUCE and r <= PrimLib.BROADCAST and dom.check_acyclic(a):
        if reduce_nums(a.ops) >= 2:
            return False
        dom_outs = set(op.output for op in dom.ops)
        a_ins = set(op_input for op in a.ops for op_input in op.inputs)
        a_outs = set(op.output for op in a.ops)
        a_final_outs = list(tensor for tensor in a_outs if tensor not in a_ins)
        stitch_tensors = list(tensor for tensor in dom_outs if tensor in a_ins)
        if not _same_stitch_axis(stitch_tensors, a_final_outs, stitch_axis_size):
            return False
        return any((tensor_size(tensor) >= stitch_buffer_size for tensor in stitch_tensors))
    return False


class CommonPattern:
    """common fuse strategies across various devices"""

    @staticmethod
    def reshape(dom):
        """fuse strategy for reshape dom"""
        if dom.pattern != PrimLib.RESHAPE:
            return []
        min_area, forward_fuse = None, False
        for a, _ in dom.out_relations.items():
            if a.pattern <= PrimLib.BROADCAST and dom.check_acyclic(a) and \
                    (min_area is None or a.pattern < min_area.pattern):
                min_area = a
        for a, _ in dom.in_relations.items():
            if a.pattern <= PrimLib.BROADCAST and a.check_acyclic(dom) and \
                    (min_area is None or a.pattern < min_area.pattern):
                min_area, forward_fuse = a, True
        return ([min_area], forward_fuse) if min_area else []

    @staticmethod
    def isolate_reshape(dom):
        """fuse strategy for isolate reshape dom"""
        if dom.pattern != PrimLib.RESHAPE or len(dom.ops) != 1:
            return []
        for a, _ in dom.out_relations.items():
            if a.mode == GraphSplitByPattern.Area.MODE_COMPOSITE and dom.check_acyclic(a):
                return [a], False
        for a, _ in dom.in_relations.items():
            if a.mode == GraphSplitByPattern.Area.MODE_COMPOSITE and a.pattern <= PrimLib.BROADCAST and \
                    a.check_acyclic(dom):
                return [a], True
        return []

    @staticmethod
    def elemwise_depth(dom):
        """fuse strategy in depth for elemwise dom"""
        if dom.pattern != PrimLib.ELEMWISE or len(dom.in_relations) != 1:
            return []
        a, r = list(dom.in_relations.items())[0]
        if a.pattern > PrimLib.ELEMWISE or len(a.out_relations) != 1 or r > PrimLib.ELEMWISE or \
                tensor_size(a.dom_op().output) != tensor_size(dom.dom_op().output):
            return []
        return [a], True

    @staticmethod
    def elemwise_width(dom):
        """fuse strategy in width for elemwise dom"""
        if dom.pattern != PrimLib.ELEMWISE:
            return []
        fused = []
        for a, r in dom.in_relations.items():
            if a.pattern <= PrimLib.ELEMWISE and r <= PrimLib.ELEMWISE and a.check_acyclic(dom) and \
                    tensor_size(a.dom_op().output) == tensor_size(dom.dom_op().output):
                fused.append(a)
        return fused, True

    @staticmethod
    def broadcast_depth(dom):
        """fuse strategy in depth for broadcast dom"""
        if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST) or len(dom.in_relations) != 1:
            return []
        a, r = list(dom.in_relations.items())[0]
        if a.pattern > PrimLib.BROADCAST or len(a.out_relations) != 1 or r > PrimLib.ELEMWISE or \
                tensor_size(a.dom_op().output) != tensor_size(dom.dom_op().output):
            return []
        return [a], True

    @staticmethod
    def broadcast_width(dom):
        """fuse strategy in width for broadcast dom"""
        if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST):
            return []
        fused = []
        for a, r in dom.in_relations.items():
            if a.pattern <= PrimLib.BROADCAST and r <= PrimLib.ELEMWISE and a.check_acyclic(dom) and \
                    tensor_size(a.dom_op().output) == tensor_size(dom.dom_op().output):
                fused.append(a)
        return fused, True

    @staticmethod
    def assign(dom):
        """fuse strategy for assign dom"""
        if len(dom.ops) != 1 or dom.dom_op().prim != "Assign":
            return []
        fused = []
        for a, _ in dom.in_relations.items():
            fused.append(a)
        return fused, True


class ReshapeElimChecker:
    """ check reshape elim """

    def __init__(self, reshape):
        def _get_remap_axis(in_shape, out_shape):
            rin, rout = [], []
            in_prod, out_prod, out_idx = 1, out_shape[-1], -1
            in_ext, out_ext = -len(in_shape) - 1, -len(out_shape) - 1
            for in_idx in range(-1, in_ext, -1):
                in_prod = in_prod * in_shape[in_idx]
                while out_prod < in_prod:
                    rout.append(out_idx)
                    out_idx = out_idx - 1
                    out_prod = out_prod * out_shape[out_idx]
                if out_prod == in_prod and out_idx > out_ext and out_shape[out_idx] == in_shape[in_idx]:
                    out_idx = out_idx - 1
                    if out_idx > out_ext:
                        out_prod = out_prod * out_shape[out_idx]
                else:
                    rin.append(in_idx)
            if out_idx > out_ext:
                rout.extend([i for i in range(out_idx, out_ext, -1)])
            return rin, rout

        remap_in, remap_out = _get_remap_axis(reshape.inputs[0].shape, reshape.output.shape)
        self.exc_fwd = self._collect_exc_ops(reshape, remap_in, True)
        self.exc_bwd = self._collect_exc_ops(reshape, remap_out, False)

    @staticmethod
    def _collect_exc_ops(reshape, remap_axis, is_fwd):
        """collect exclude ops of reshape"""

        def _propagate(remap, src, des):
            out_remap = []
            src_prod, des_prod, des_idx = 1, 1, 0
            for src_idx in range(-1, -len(src) - 1, -1):
                src_prod = src_prod * src[src_idx]
                if src_idx in remap:
                    while des_prod < src_prod:
                        des_idx = des_idx - 1
                        des_prod = des_prod * des[des_idx]
                        out_remap.append(des_idx)
                else:
                    while des_prod < src_prod:
                        prod = des_prod * des[des_idx - 1]
                        if prod > src_prod:
                            break
                        des_idx, des_prod = des_idx - 1, prod
            return out_remap

        def _remap_check(op, remap, iter_type):
            if iter_type not in (PrimLib.ELEMWISE, PrimLib.BROADCAST):
                return False
            for t in op.inputs:
                for i in remap:
                    if -i <= len(t.shape) and t.shape[i] != op.output.shape[i]:
                        return False
            return True

        def push_stack(op, remap):
            stack.append((op, remap))
            visited.add(op)

        def _visit_fwd(op, remap):
            for t in op.inputs:
                if t.op is None:
                    _visit_bwd(t, remap)
                elif tensor_size(t) > 1 and t.op not in visited:  # all broadcast
                    iter_type = PrimLib.iter_type(t.op)
                    if iter_type == PrimLib.RESHAPE:
                        new_remap = _propagate(remap, t.shape, t.op.inputs[0].shape)
                        push_stack(t.op, new_remap)
                    elif _remap_check(t.op, remap, iter_type):
                        push_stack(t.op, remap)
                    else:
                        exc_ops.add(t.op)

        def _visit_bwd(t, remap):
            for op in t.to_ops:
                if op not in visited:
                    iter_type = PrimLib.iter_type(op)
                    if iter_type == PrimLib.REDUCE and tensor_size(op.output) == 1:  # all reduce
                        continue
                    if iter_type == PrimLib.RESHAPE:
                        new_remap = _propagate(remap, t.shape, op.output.shape)
                        push_stack(op, new_remap)
                    elif _remap_check(op, remap, iter_type):
                        push_stack(op, remap)
                    else:
                        exc_ops.add(op)

        exc_ops, stack, visited = set(), [], {reshape}
        if is_fwd:
            _visit_fwd(reshape, remap_axis)
        else:
            _visit_bwd(reshape.output, remap_axis)
        while stack:
            top, remap = stack.pop()
            _visit_bwd(top.output, remap)
            _visit_fwd(top, remap)
        return exc_ops

    def check(self, ops, is_fwd):
        """ fuse check """
        if is_fwd:
            fwd_res = all([op not in self.exc_fwd for op in ops]) if self.exc_fwd is not None else False
            bwd_res = self.exc_bwd is not None
        else:
            fwd_res = self.exc_fwd is not None
            bwd_res = all([op not in self.exc_bwd for op in ops]) if self.exc_bwd is not None else False
        return [fwd_res, bwd_res] if fwd_res or bwd_res else False

    def commit(self, res):
        """ commit fuse result """
        if not res[0] and self.exc_fwd is not None:
            self.exc_fwd = None
        if not res[1] and self.exc_bwd is not None:
            self.exc_bwd = None


class ReduceOutFuseChecker:
    """Reduce output fuse checker """

    def __init__(self, red_op):
        self.output_excluded = set()
        recursion_stack = [red_op]
        while recursion_stack:
            op = recursion_stack.pop()
            for to in op.output.to_ops:
                idx = to.inputs.index(op.output)
                if PrimLib.iter_type(to) > PrimLib.ELEMWISE or \
                        tensor_size(to.inputs[idx]) != tensor_size(to.output):
                    self.output_excluded.add(to)
                else:
                    recursion_stack.append(to)

    def check(self, ops, is_fwd):
        """ fuse check """
        if not is_fwd and self.output_excluded:
            for op in self.output_excluded:
                if op in ops:
                    return False
        return True

    def commit(self, res):
        """ commit fuse result """
        del res
        return self.output_excluded  # I'm not static


class GraphSplitByPattern:
    """Graph splitter"""

    class ReachTable:
        """Reachable table"""

        def __init__(self, size):
            self.map = []
            self.alive = set(range(size))
            for i in range(0, size):
                self.map.append([False] * size)
                self.map[i][i] = True

        def reachable(self, x, y):
            """reachable from x to y"""
            return self.map[x][y]

        def sync(self, x, y):
            """sync from y to x"""
            for i in self.alive:
                self._link(self.map[y][i], x, i)

        def _link(self, cond, f, t):
            """link from `f` to `t`"""
            if cond:
                self.map[f][t] = True

        def fuse(self, x, y):
            """fuse y to x"""
            for i in self.alive:
                # i is the succeeding node of y, links the x's previous nodes to i
                if self.map[y][i] and not self.map[x][i]:
                    for pre in self.alive:
                        self._link(self.map[pre][x], pre, i)
                # i is the previous node of y, link i to x's succeeding nodes
                if self.map[i][y] and not self.map[i][x]:
                    for suc in self.alive:
                        self._link(self.map[x][suc], i, suc)
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

            def has_stitch_op(self):
                """check stitch_op exists"""
                return self.stitch_ops or self.stitch_atomic_ops

        def __init__(self, init_op, is_output, unique_id, reach_tab):
            self.pattern = PrimLib.iter_type(init_op) if init_op is not None else PrimLib.UNKNOWN
            self.ops = [] if init_op is None else [init_op]
            self.in_relations = dict()  # {area1: relation1, area2: relation2, ...}
            self.out_relations = dict()  # {area1: relation1, area2: relation2, ...}
            self.mode = None
            self.stitch_info = self.StitchInfo()
            self.recompute_ops = []
            self.is_output = is_output
            self.output_excluded = set()
            self.unique_id = unique_id
            self.reach_tab = reach_tab
            self.checkers = []
            if self.pattern == PrimLib.RESHAPE and init_op.inputs:  # reshape's input may be empty (const value)
                self.checkers.append(ReshapeElimChecker(init_op))
            elif self.pattern == PrimLib.REDUCE:
                self.checkers.append(ReduceOutFuseChecker(init_op))

        def __str__(self):
            return '<' + '-'.join((op.output.name for op in self.ops)) + '>'

        def __repr__(self):
            return str(self)

        @staticmethod
        def get_relation(op, i):
            """Get op relation"""
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
            """Update stitch info"""
            if stitch_info.stitch_ops:
                self.stitch_info.stitch_ops.update(stitch_info.stitch_ops)
            if stitch_info.stitch_atomic_ops:
                self.stitch_info.stitch_atomic_ops.update(stitch_info.stitch_atomic_ops)

        def fuse_confirm(self, area):
            """confirm if area can be fused"""

            def _check(a, b, res, fwd):
                for checker in a.checkers:
                    r = checker.check(b.ops, fwd)
                    if not r:
                        return False
                    res.append(r)
                return True

            def _commit(a, res):
                for i, checker in enumerate(a.checkers):
                    checker.commit(res[i])

            res1, res2 = [], []
            if not _check(self, area, res1, True) or not _check(area, self, res2, False):
                return False
            _commit(self, res1)
            _commit(area, res2)
            return True

        def fuse_prepare(self, dom):
            """do some prepare before fused to dom"""
            del dom
            return self.unique_id  # I'm not static method

        def fuse_done(self, dom):
            """do some thing after fused to dom"""
            dom.reach_tab.fuse(dom.unique_id, self.unique_id)

        def fuse(self, area):
            """Fuse `area` to `self`"""

            def _update_relation(relations, a, r):
                relations[a] = max(r, relations[a]) if a in relations else r

            def _update_pattern():
                if area.pattern > self.pattern:
                    self.pattern = area.pattern
                if area in self.in_relations and self.in_relations.get(area) > self.pattern:
                    self.pattern = self.in_relations.get(area)

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

            area.fuse_prepare(self)
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
            self.update_stitch_info(area.stitch_info)
            self.recompute_ops.extend(area.recompute_ops)
            self.checkers.extend(area.checkers)
            area.fuse_done(self)

        def check_acyclic(self, to):
            """Check circle. It returns false if circle exists"""
            for out, _ in self.out_relations.items():
                if out != to and self.reach_tab.reachable(out.unique_id, to.unique_id):
                    return False
            return True

        def dom_op(self):
            """Get dom op"""
            return self.ops[0]

    class RecomputeArea(Area):
        """RecomputeArea"""

        def __init__(self, unique_id, reach_tab):
            super().__init__(None, False, unique_id, reach_tab)
            self.recom_pre = None
            self.recom_user = None
            self.recom_dom = None
            self.dom_user_r = PrimLib.UNKNOWN
            self.ori_op_map = {}
            self.recom_map = {}
            self.fuse_success = False

        def fuse_prepare(self, dom):
            """copy recompute_ops in area to ops, self is area's user"""
            tail_tensor = self.recompute_ops[-1].output
            # copy tensors, all copied are Tensor.PARA_NONE
            tensor_map = {}
            if self.recompute_ops[0].inputs:
                tensor_map[self.recompute_ops[0].inputs[0]] = self.recompute_ops[0].inputs[0]
            for op in self.recompute_ops:
                orig_tensor = op.output
                cp_tensor = Tensor(orig_tensor.name, orig_tensor.shape, orig_tensor.dtype, orig_tensor.data_format)
                tensor_map[orig_tensor] = cp_tensor
            # copy ops
            cp_ops = []
            for op in self.recompute_ops:
                inputs = [tensor_map.get(op.inputs[0])] if op.inputs else []
                cp_op = Operator(op.prim, inputs, tensor_map.get(op.output), op.attrs)
                cp_op.all_inputs = cp_op.inputs
                cp_ops.append(cp_op)
                self.ori_op_map[cp_op] = op
            # connect copied ops
            for op in dom.ops:
                if tail_tensor in op.inputs:
                    op.inputs.remove(tail_tensor)
                    op.inputs.append(tensor_map.get(tail_tensor))
                    tail_tensor.to_ops.remove(op)
                    tensor_map.get(tail_tensor).to_ops.append(op)
            # fill cp_ops in self.recompute_area
            cp_dom_op = None
            for cp, ori in self.ori_op_map.items():
                if ori == self.dom_op():
                    cp_dom_op = cp
            self.ops.clear()
            self.ops.append(cp_dom_op)
            self.ops.extend((op for op in cp_ops if op != cp_dom_op))

        def fuse_done(self, dom):
            """do some thing after fused to dom"""
            del dom
            self.fuse_success = True

        def reset(self, dom_area, ops, user_area, pre_area):
            """set the recompute area and connect with other areas"""
            self.recompute_ops.extend(ops)
            # recom_area: set dom_op and correct ops length
            patterns = list(PrimLib.iter_type(op) for op in ops)
            self.pattern = max(patterns)
            for i, pat in enumerate(patterns):
                if pat == self.pattern:
                    self.ops = [ops[i]] * len(ops)
                    break
            # disconnect dom_area and user_area
            self.dom_user_r = dom_area.out_relations[user_area]
            dom_area.out_relations.pop(user_area)
            user_area.in_relations.pop(dom_area)
            # connect recom_area and user_area
            user_area.in_relations[self] = self.dom_user_r
            self.out_relations[user_area] = self.dom_user_r
            # connect recom_pre and recom_area
            self.recom_pre = pre_area
            if self.recom_pre is not None:
                self.in_relations[self.recom_pre] = dom_area.in_relations[self.recom_pre]
                self.recom_pre.out_relations[self] = dom_area.in_relations[self.recom_pre]
            # set related areas
            self.recom_user = user_area
            self.recom_dom = dom_area
            self.fuse_success = False

        def clear(self):
            """disconnect recom_area from other areas, and clear recom_area"""
            self.out_relations.clear()
            self.in_relations.clear()
            if not self.fuse_success:
                self.recom_user.in_relations.pop(self)
                self.recom_user.in_relations[self.recom_dom] = self.dom_user_r
                self.recom_dom.out_relations[self.recom_user] = self.dom_user_r
                if self.recom_pre:
                    self.recom_pre.out_relations.pop(self)
            self.ops.clear()
            self.recompute_ops.clear()
            self.recom_map.update(self.ori_op_map)
            self.ori_op_map.clear()

    def __init__(self, graph, flags):
        self.graph = graph
        self.areas = []
        self.flags = flags
        self.enable_recompute = self.flags.get("enable_recompute_fusion", False)
        self.enable_stitch_fusion = self.flags.get("enable_stitch_fusion", False)
        self.enable_horizontal_fusion = self.flags.get("enable_horizontal_fusion", False)
        self.reduce_fuse_depth = self.flags.get("reduce_fuse_depth", -1)
        self.reach_tab = self.ReachTable(len(graph.ops) + 1 if self.enable_recompute else len(graph.ops))
        self.area_map = {}
        _, outputs = graph.deduce_parameters()
        idx = 0
        for op in graph.ops:
            is_output = op.output in outputs
            a = self.Area(op, is_output, idx, self.reach_tab)
            idx += 1
            self.set_default_mode(a)
            self.areas.append(a)
            self.set_area_map([op], a)
        for a in self.areas:
            a.link_input(self.area_map)
        for i in range(len(self.areas) - 1, -1, -1):
            self.areas[i].link_output()
        if self.enable_recompute:
            self.recom_area = self.RecomputeArea(idx, self.reach_tab)

    def set_area_map(self, ops, area):
        """update area_map after op fused to area"""
        for op in ops:
            self.area_map[op] = area

    def set_default_mode(self, area):
        """Set default mode"""
        area.mode = self.get_default_mode(area.ops[0])

    @staticmethod
    def limit_area_size(dominant, fuse_areas, limit_size):
        """Remove some areas if the size is too large"""
        area_sizes = map(lambda area: len(area.ops), fuse_areas)
        dom_size = len(dominant.ops)
        if dom_size + prod_reduce(lambda x, y: x + y, area_sizes) <= limit_size:
            return fuse_areas
        # fuse the smaller area in priority
        fuse_areas.sort(key=lambda area: len(area.ops))
        new_fuse_areas = []
        for area in fuse_areas:
            if dom_size + len(area.ops) > limit_size:
                break
            dom_size += len(area.ops)
            new_fuse_areas.append(area)
        return new_fuse_areas

    def fuse(self, selector, is_stitch=False):
        """Fuse areas"""

        def _fuse_area():
            for dominant in self.areas:
                result = selector(dominant)
                if not result or not result[0]:
                    continue
                fuse_areas, is_forward = result
                if not is_stitch:
                    fuse_areas = self.limit_area_size(dominant, fuse_areas, self.flags['composite_op_limit_size'])
                    if not fuse_areas:
                        continue
                changed = False
                if is_forward:
                    for area in fuse_areas:
                        if is_stitch or dominant.fuse_confirm(area):
                            dominant.fuse(area)
                            self.set_area_map(area.ops, dominant)
                            self.areas.remove(area)
                            changed = True
                else:
                    forward_area = dominant
                    for area in fuse_areas:
                        if is_stitch or area.fuse_confirm(forward_area):
                            area.fuse(forward_area)
                            self.set_area_map(forward_area.ops, area)
                            self.areas.remove(forward_area)
                            forward_area = area
                            changed = True
                if changed:
                    return True
            return False

        changed, do_again = False, True
        while do_again:
            do_again = _fuse_area()
            changed = changed or do_again
        return changed

    def hfuse(self, selector):
        """Fuse horizontal areas with same input tensor"""

        def _do_fuse(areas):
            for i in range(len(areas) - 1):
                dom = areas[i]
                for a in areas[i + 1:]:
                    if dom.check_acyclic(a) and a.check_acyclic(dom) and \
                            selector(dom, a) and self.limit_area_size(dom, [a], 64) and dom.fuse_confirm(a):
                        dom.fuse(a)
                        self.set_area_map(a.ops, dom)
                        self.areas.remove(a)
                        return True
            return False

        def _update_areas(areas, from_op):
            for op in from_op.to_ops:
                a = self.area_map.get(op)
                if a in self.areas and a not in areas:
                    areas.append(a)

        changed = False
        while True:
            for dom in self.areas:
                if len(dom.out_relations) > 1 and _do_fuse(list(dom.out_relations.keys())):
                    changed = True
                    break
            else:
                break
        inputs, _ = self.graph.deduce_parameters()
        while True:
            for t in inputs:
                areas = []
                _update_areas(areas, t)
                if len(areas) > 1 and _do_fuse(areas):
                    changed = True
                    break
            else:
                break
        return changed

    def fuse_recom(self, selector):
        """Fuse recompute area to its user"""
        user = self.recom_area.recom_user
        for dominant in [self.recom_area, user]:
            result = selector(dominant)
            if result and result[0]:
                fuse_areas, _ = result
                fuse_areas = self.limit_area_size(dominant, fuse_areas, self.flags['composite_op_limit_size'])
                if not fuse_areas:
                    continue
                if fuse_areas[0] in [self.recom_area, user] and user.fuse_confirm(self.recom_area):
                    user.fuse(self.recom_area)
                    self.set_area_map(self.recom_area.ops, user)
                    return True
        return False

    def index_op(self):
        """index op by order, the copied op share id with original op, for topo-sort"""
        ids = {}
        for i, op in enumerate(self.graph.ops):
            ids[op] = i
        if self.enable_recompute:
            for k, v in self.recom_area.recom_map.items():
                ids[k] = ids.get(v)
        return ids

    def to_subgraphs(self):
        """Transform op groups to subgraphs"""
        ids = self.index_op()
        subgraphs = []
        graphmodes = []
        for i, area in enumerate(self.areas):
            area.ops.sort(key=ids.get)
            subgraphs.append(Graph('{}_{}'.format(self.graph.name, i), area.ops, area.stitch_info, area.recompute_ops))
            graphmodes.append("basic" if area.mode == self.Area.MODE_BASIC else "composite")
        return subgraphs, graphmodes

    def pattern_fuse(self, fuse_func=None):
        """fuse Areas by pattern repeatedly"""
        del fuse_func
        raise Exception("pattern_fuse() is not implemented in {}".format(self.__class__.__name__))

    def split(self):
        """Split graph by pattern"""
        self.pattern_fuse()
        if self.enable_recompute:
            self.recompute_fuse()
        # The reshape should not be output node
        # Note: after this function, the input output relation is not maintained.
        self.split_output_reshapes()
        subgraphs, graphmodes = self.to_subgraphs()
        return subgraphs, graphmodes

    def split_output_reshapes(self):
        """Force split the output Reshapes into other new area"""

        def _remove_output_reshape(reshape_ops, other_ops):
            def _run():
                for op in reshape_ops:
                    if any((to_op in other_ops for to_op in op.output.to_ops)):
                        reshape_ops.remove(op)
                        other_ops.append(op)
                        return True
                return False

            while _run():
                pass

        new_areas = []
        for area in self.areas:
            reshape_ops = list(op for op in area.ops if PrimLib.iter_type(op) == PrimLib.RESHAPE)
            other_ops = list(op for op in area.ops if op not in reshape_ops)
            if not other_ops or not reshape_ops:
                continue
            # remove the output reshape from "reshape_ops" and add it into "other_ops"
            _remove_output_reshape(reshape_ops, other_ops)
            if not reshape_ops:
                continue
            for op in reshape_ops:
                a = self.Area(op, False, 0, self.reach_tab)
                self.set_default_mode(a)
                new_areas.append(a)
            area.ops = other_ops
            if len(other_ops) == 1:
                self.set_default_mode(area)
        if new_areas:
            self.areas += new_areas

    def recompute_fuse(self):
        """find recompute regions and copy them out to new Areas"""

        def _get_prods(area, border):
            """get producer region of border op"""
            max_weight = 10
            stack = [border]
            ops, inputs = [], []
            while stack:
                op = stack.pop()
                if len(op.inputs) > 1 or PrimLib.iter_type(op) > PrimLib.BROADCAST or len(ops) > max_weight:
                    return []
                ops.append(op)
                for t in op.inputs:
                    if t.op in area.ops:
                        stack.append(t.op)
                    else:
                        inputs.append(t)
            return ops, inputs

        def _get_border_info(area):
            """get border information"""
            prods, users = {}, {}
            for op in area.ops:
                if len(op.output.to_ops) <= 1 and op.output.para_type != Tensor.PARA_OUTPUT:
                    continue
                for to in op.output.to_ops:
                    if to in area.ops:
                        continue
                    user = self.area_map.get(to)
                    if user.pattern > PrimLib.RESHAPE:
                        if user in users:
                            users.get(user).append(op)
                        else:
                            users[user] = [op]
                        if op not in prods:
                            prods[op] = _get_prods(area, op)
            return prods, users

        def _get_cheap_region(prods, borders):
            """get cheap region of border ops"""
            if len(borders) > 1:
                return []
            result = []
            for op in borders:
                if prods[op]:
                    prod_ops, inputs = prods[op]
                    if sum([t.get_size() for t in inputs]) <= op.output.get_size():
                        pred = self.area_map.get(inputs[0].op) if inputs and inputs[0].op else None
                        result.append([pred, prod_ops[::-1]])
            return result

        def _do_recompute(area):
            """split the unfusing pattern by add recompute area"""
            prods, users = _get_border_info(area)
            for user, borders in users.items():
                result = _get_cheap_region(prods, borders)
                for pred, region in result:
                    self.recom_area.reset(area, region, user, pred)
                    self.pattern_fuse(self.fuse_recom)
                    self.recom_area.clear()
                    if self.recom_area.fuse_success:
                        return True
            return False

        changed = True
        while changed:
            changed = False
            orig_areas = []
            orig_areas.extend(self.areas)
            for area in orig_areas:
                if area in self.areas and area.out_relations:
                    changed = _do_recompute(area) or changed
            if changed:
                self.pattern_fuse()


class GraphSplitGpu(GraphSplitByPattern):
    """Graph splitter"""
    BROADCAST_FUSE_DEPTH = 20
    TRANSPOSE_FUSE_DEPTH = 6

    def __init__(self, graph, flags):
        super().__init__(graph, flags)
        self.reduce_fuse_depth = 20 if self.reduce_fuse_depth < 0 else self.reduce_fuse_depth

    def get_default_mode(self, op):
        """Get default mode in GPU"""
        if op.prim == "MatMul":
            return self.Area.MODE_COMPOSITE if op.inputs[0].dtype == "float16" and op.attrs['Akg'] else \
                self.Area.MODE_BASIC
        if op.prim == "Assign":
            return self.Area.MODE_BASIC
        pattern = PrimLib.iter_type(op)
        return self.Area.MODE_BASIC if pattern == PrimLib.RESHAPE else self.Area.MODE_COMPOSITE

    def pattern_fuse(self, fuse_func=None):
        """fuse Areas by pattern"""

        def _broadcast_pat_exclude(dom, a, r):
            if a.pattern == PrimLib.REDUCE:
                return dom.pattern > PrimLib.ELEMWISE or r > PrimLib.ELEMWISE
            return a.pattern > PrimLib.REDUCE or r > PrimLib.BROADCAST

        def _broadcast_bwd_depth(dom):
            if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST) or len(dom.out_relations) != 1 or \
                    dom.is_output or len(dom.ops) > self.BROADCAST_FUSE_DEPTH:
                return []
            a, r = list(dom.out_relations.items())[0]
            if _broadcast_pat_exclude(dom, a, r) or len(a.in_relations) != 1:
                return []
            return [a], False

        def _broadcast_bwd_width(dom):
            if dom.pattern not in (PrimLib.ELEMWISE, PrimLib.BROADCAST) or \
                    dom.is_output or len(dom.ops) > self.BROADCAST_FUSE_DEPTH:
                return []
            fused = []
            for a, r in dom.out_relations.items():
                if _broadcast_pat_exclude(dom, a, r) or not dom.check_acyclic(a) or \
                        (fused and tensor_size(fused[0].dom_op().output) != tensor_size(a.dom_op().output)):
                    return []
                fused.append(a)
            return fused, False

        def _reduce_pat_exclude(_, a, r):
            if len(a.ops) > self.reduce_fuse_depth:
                return True
            return a.pattern > PrimLib.ELEMWISE or r > PrimLib.REDUCE or r == PrimLib.BROADCAST

        def _reduce_depth(dom):
            if dom.pattern != PrimLib.REDUCE or len(dom.in_relations) != 1:
                return []
            a, r = list(dom.in_relations.items())[0]
            if dom.ops[0].inputs[0].dtype == "float16" and a.is_output and len(a.ops) >= 10 and \
                    _is_atomic_add_available(dom):
                # to evade the precision problem.
                return []
            if _reduce_pat_exclude(dom, a, r) or len(a.out_relations) != 1:
                return []
            return [a], True

        def _reduce_width(dom):
            if dom.pattern != PrimLib.REDUCE:
                return []
            fused = []
            for a, r in dom.in_relations.items():
                if dom.ops[0].inputs[0].dtype == "float16" and a.is_output and len(a.ops) >= 10 and \
                        _is_atomic_add_available(dom):
                    # to evade the precision problem.
                    continue
                if not _reduce_pat_exclude(dom, a, r) and a.check_acyclic(dom):
                    fused.append(a)
            return fused, True

        def _is_atomic_add_available(dom):
            if any(("Reduce" in x.prim for x in dom.ops[1:])):
                return False
            op = dom.ops[0]
            if "reduce_axis" in op.attrs:
                reduce_axis = op.attrs["reduce_axis"]
            elif "axis" in op.attrs:
                reduce_axis = [op.attrs["axis"]]
            else:
                raise Exception("For '{}', can not find the attr 'reduce_axis' or 'axis'".format(op.prim))
            if op.inputs and len(op.inputs[0].shape) - 1 in reduce_axis:
                reduce_size = prod_reduce(lambda x, y: x * y, (op.inputs[0].shape[i] for i in reduce_axis))
                return reduce_size >= 1024
            return True

        def _may_multi_filter(dom_ops):
            count = 1
            stack = [dom_ops[0]]
            while stack:
                op = stack.pop()
                for t in op.inputs:
                    if t.op and t.op in dom_ops:
                        count = count + 1
                        stack.append(t.op)
            return count < len(dom_ops)

        def _reduce_output(dom):
            if dom.pattern != PrimLib.REDUCE:
                return []
            if _may_multi_filter(dom.ops):
                return []
            if _is_atomic_add_available(dom):
                return []
            is_all_reduce = tensor_size(dom.ops[0].output) == 1
            # excluded large size all reduce
            if is_all_reduce and dom.ops[0].inputs and tensor_size(dom.ops[0].inputs[0]) > 1024 * 12:
                return []

            fused = []
            for a, r in dom.out_relations.items():
                if a.pattern <= PrimLib.BROADCAST and r <= PrimLib.BROADCAST and dom.check_acyclic(a):
                    fused.append(a)
            return fused, False

        def _reduce_stitch(dom):
            if dom.pattern != PrimLib.REDUCE:
                return []
            if tensor_size(dom.ops[0].output) == 1:
                return []
            if tensor_size(dom.ops[0].inputs[0]) < 1024 * 12:
                return []

            fused = []
            for a, r in dom.out_relations.items():
                if not may_stitch(dom, a, r, 1024 * 8, 1024 * 1024):
                    continue
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
                return []
            fused = []
            for a, _ in dom.in_relations.items():
                if a.pattern <= PrimLib.BROADCAST and a.check_acyclic(dom) and len(a.ops) <= self.TRANSPOSE_FUSE_DEPTH:
                    fused.append(a)
            return fused, True

        def _strided_slice(dom):
            if dom.dom_op().prim != "StridedSlice":
                return []
            fused = []
            for a, _ in dom.in_relations.items():
                if a.pattern <= PrimLib.BROADCAST and a.check_acyclic(dom) and \
                        len(a.out_relations) == 1 and not a.is_output:
                    fused.append(a)
            return fused, True

        def _gather_output(dom, reduce_fusion=False):
            gather_prims = ("Gather", "GatherNd", "CSRGather")
            if not dom.dom_op().prim in gather_prims:
                return []

            def _reduce_exclude(op, axis_list):
                """ Whether this operator should be excluded.
                Excluding condition:
                1. There are at least one same axis between reduce axes and axis_list.

                Args:
                    op (Operator): Target reduce operator.
                    axis_list (list): List to check whether it is intersected by reduce axis.
                Returns:
                    Boolean. Whether this operator should be excluded.
                """
                axis = op.attrs["reduce_axis"]
                if isinstance(axis, int):
                    axis = [axis]
                in_shape_len = len(op.inputs[0].shape)
                for i, dim in enumerate(axis):
                    axis[i] = in_shape_len + dim if dim < 0 else dim
                fix_axis = []
                for ax in axis:
                    if op.inputs[0].shape[ax] == 1:
                        continue
                    fix_axis.append(ax)
                return bool(set(fix_axis) & set(axis_list))

            def _bfs_visit(start_op, start_prims, total_ops, end_ops, gather_axis):
                consisten_shape = start_op.output.shape
                visited = []
                op_queue = [start_op]

                def _early_stop(cur_op):
                    if cur_op in end_ops:
                        # If reduce the gather axis, stop early for not fusion.
                        if cur_op.prim == "ReduceSum" and _reduce_exclude(cur_op, gather_axis):
                            return True
                    else:
                        if (cur_op.prim in start_prims and cur_op != start_op) or \
                                consisten_shape != cur_op.output.shape:
                            return True
                    return False

                while op_queue:
                    tmp_queue = []
                    for op in op_queue:
                        if op in visited or op not in total_ops:
                            continue
                        if _early_stop(op):
                            return False
                        if op in end_ops:
                            continue
                        for to_op in op.output.to_ops:
                            tmp_queue.append(to_op)
                        visited.append(op)
                    op_queue = tmp_queue
                return True

            def _shape_consistent(start_prims, end_prims, source, target):
                """
                Check whether it is always shape consistent from source nodes to target nodes.
                Excluding condition:
                    When fusing ReduceSum, first check if TensorScatterAdd and/or UnsortedSegmentSum
                    has already been fused, if so, stop ReduceSum fusion.
                """
                total_ops = source.ops + target.ops
                op_prims_set = {op.prim for op in total_ops}
                if reduce_fusion and (len({"TensorScatterAdd", "UnsortedSegmentSum"} & op_prims_set) >= 1):
                    return False
                start_ops = []
                for op in source.ops:
                    if op.prim in start_prims:
                        start_ops.append(op)
                end_ops = []
                for op in total_ops:
                    if op.prim in end_prims and not any((to_op in total_ops for to_op in op.output.to_ops)):
                        end_ops.append(op)

                for start_op in start_ops:
                    gather_axis = start_op.attrs.get("axis", None)
                    if gather_axis is None:
                        # For GatherNd
                        gather_axis = list(range(len(start_op.inputs[1].shape)))
                    elif isinstance(gather_axis, int):
                        gather_axis = [gather_axis]

                    is_consistent = _bfs_visit(start_op, start_prims, total_ops, end_ops, gather_axis)
                    if not is_consistent:
                        return False
                return True

            if reduce_fusion:
                appected_areas = {"ReduceSum", "CSRReduceSum"}
            else:
                appected_areas = {"TensorScatterAdd", "UnsortedSegmentSum"}

            for a, _ in dom.out_relations.items():
                if _shape_consistent(gather_prims, appected_areas, dom, a) and dom.check_acyclic(a):
                    return [a], False
            return []

        def _broadcast_tot(dom):
            """Fuse rule for TensorScatterAdd and UnsortedSegmentSum."""

            def _same_input(op1, op2):
                return bool(set(op1.inputs) & set(op2.inputs))

            if len(dom.ops) != 1:
                return []

            # Only fuse the first input for `TensorScatterAdd`` and the first and second input for `UnsortedSegmentSum`.
            fuse_arg = {"TensorScatterAdd": slice(1, None), "UnsortedSegmentSum": slice(0, 2)}
            arg_idx = fuse_arg.get(dom.dom_op().prim, -1)
            if arg_idx == -1:
                return []
            fuse_tensor = dom.dom_op().inputs[arg_idx]

            for a, _ in dom.in_relations.items():
                if not a.check_acyclic(dom):
                    continue
                # Rule 1: Same type with at lease one same input.
                if a.dom_op().prim == dom.dom_op().prim and _same_input(dom.dom_op(), a.dom_op()):
                    return [a], True
                # Rule 2: Fuse op(reshape/elementwise/broadcast) in specified position inputs.
                if a.pattern <= PrimLib.BROADCAST and any((op.output in fuse_tensor for op in a.ops)):
                    return [a], True
            return []

        def _broadcast_onehot(dom, fwd=True):
            """Fuse rule for OneHot."""
            if dom.dom_op().prim != "OneHot":
                return []

            fused = []
            neighbours = dom.in_relations.items() if fwd else dom.out_relations.items()
            for a, _ in neighbours:
                if a.pattern <= PrimLib.BROADCAST:
                    if (fwd and a.check_acyclic(dom) and len(a.out_relations) == 1 and not a.is_output) or \
                            (not fwd and dom.check_acyclic(a)):
                        fused.append(a)

            return fused, fwd

        def _elemwise_elemany(dom):
            """Fuse rule for elemany."""
            if dom.dom_op().prim != "ElemAny":
                return []

            fused = []
            for a, r in dom.in_relations.items():
                if a.pattern < PrimLib.BROADCAST and r <= PrimLib.ELEMWISE and a.check_acyclic(dom):
                    fused.append(a)

            return fused, True

        def _injective_output(dom):
            """Fuse rule for injective """
            injective_ops = {"Transpose", "StridedSlice"}
            if dom.dom_op().prim not in injective_ops:
                return []
            to_ops = dom.dom_op().output.to_ops
            if dom.is_output or len(to_ops) != 1 or len(dom.out_relations) != 1:
                return []
            to_area = list(dom.out_relations.keys())[0]
            if (to_area.pattern >= PrimLib.REDUCE and to_area.dom_op().prim not in injective_ops) or \
                    to_ops[0] not in to_area.ops:
                return []
            if len(to_area.ops) > self.TRANSPOSE_FUSE_DEPTH:
                return []
            return [to_area], False

        def _h_broadcast(dom, a):
            if dom.pattern > PrimLib.BROADCAST:
                return []
            return a.pattern <= PrimLib.BROADCAST and dom.ops[0].output.shape == a.ops[0].output.shape

        def _h_reduce(dom, a):
            if dom.pattern != PrimLib.REDUCE or dom.stitch_info.stitch_ops:
                return []
            dom_op = dom.ops[0]
            if not PrimLib.is_reduce(dom_op) or _is_atomic_add_available(dom):
                return []
            op = a.ops[0]
            return a.pattern == PrimLib.REDUCE and not a.stitch_info.stitch_ops and \
                PrimLib.is_reduce(op) and dom_op.inputs[0].shape == op.inputs[0].shape and \
                dom_op.attrs.get("reduce_axis") == op.attrs.get("reduce_axis")

        def _h_opaque(dom, a):
            if dom.ops[0].prim not in {"StridedSlice"}:
                return []
            return a.ops[0].prim == dom.ops[0].prim and dom.ops[0].output.shape == a.ops[0].output.shape and \
                dom.ops[0].inputs[0].shape == a.ops[0].inputs[0].shape

        def _link_csr(dom):
            def _same_input(op1, op2):
                return bool(set(op1.inputs.copy()) & set(op2.inputs.copy()))

            fuse_arg = {"CSRReduceSum": slice(1, 3), "CSRGather": slice(2, 3)}
            arg_idx = fuse_arg.get(dom.dom_op().prim, -1)
            if arg_idx == -1:
                return []
            fuse_tensor = dom.dom_op().inputs[arg_idx]
            for a, _ in dom.in_relations.items():
                if (a.dom_op().prim == "CSRGather" and a.dom_op().prim == dom.dom_op().prim and
                        _same_input(dom.dom_op(), a.dom_op())):
                    return [a], True
                if a.pattern <= PrimLib.BROADCAST and dom.check_acyclic(a) and \
                        any([op.output in fuse_tensor for op in a.ops]):
                    return [a], True
            return []

        def _fuse_loop():
            self.fuse(CommonPattern.reshape)
            self.fuse(CommonPattern.assign)
            self.fuse(CommonPattern.elemwise_depth)
            self.fuse(CommonPattern.elemwise_width)
            self.fuse(_broadcast_tot)
            self.fuse(_link_csr)
            self.fuse(CommonPattern.broadcast_depth)
            self.fuse(CommonPattern.broadcast_width)
            self.fuse(_reduce_depth)
            self.fuse(_reduce_width)
            self.fuse(_broadcast_bwd_depth)
            self.fuse(_broadcast_bwd_width)
            self.fuse(_strided_slice)
            self.fuse(partial(_broadcast_onehot, fwd=True))
            self.fuse(partial(_broadcast_onehot, fwd=False))
            self.fuse(partial(_gather_output, reduce_fusion=False))
            self.fuse(partial(_gather_output, reduce_fusion=True))
            self.fuse(_reduce_output)
            if self.enable_stitch_fusion:
                self.fuse(_reduce_stitch, True)
            self.fuse(_transpose)
            self.fuse(_injective_output)
            self.fuse(CommonPattern.isolate_reshape)
            if self.enable_horizontal_fusion:
                self.hfuse(_h_broadcast)
                self.hfuse(_h_reduce)
                self.hfuse(_h_opaque)
            self.fuse(_elemwise_elemany)

        def _fuse_once(fuse_func):
            if fuse_func(CommonPattern.reshape) or \
                    fuse_func(CommonPattern.elemwise_depth) or fuse_func(CommonPattern.elemwise_width) or \
                    fuse_func(CommonPattern.broadcast_depth) or fuse_func(CommonPattern.broadcast_width) or \
                    fuse_func(_reduce_depth) or fuse_func(_reduce_width) or \
                    fuse_func(_broadcast_bwd_depth) or fuse_func(_broadcast_bwd_width):
                return
            if fuse_func(_reduce_output):
                return
            fuse_func(_transpose)

        if fuse_func is None:
            _fuse_loop()
        else:
            _fuse_once(fuse_func)


def split(graph, target, flags):
    """Split graph"""
    result = None
    if target == "cuda":
        result = GraphSplitGpu(graph, flags).split()
    return result
