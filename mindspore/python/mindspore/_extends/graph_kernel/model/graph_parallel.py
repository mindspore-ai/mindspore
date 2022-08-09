# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Cost model for parallel fusion"""
from __future__ import division
from .model import PrimLib


class ParalGain:
    """Paral Gain"""

    def __init__(self, fusion_type, bottleneck, gain, block_assign, type_info):
        self.fusion_type = fusion_type
        self.bottleneck = bottleneck
        self.gain = gain
        self.block_assign = block_assign
        self.type_info = type_info


class ScheduleAnalyzer:
    """schedule analyzer"""
    WRAP_SIZE = 32
    MAX_SM = 80  # Volta
    MAX_NUM_THREADS = 1024
    MAX_BLOCK = 256
    PIPELINE_OP_THREADHOLD = 5

    def __init__(self, graph):
        self.graph = graph
        self.block_num = 0
        self.block_weight = 0
        _, outputs = graph.deduce_parameters()
        self.ops = graph.ops
        self.dom_op = list(out.op for out in outputs)

    @staticmethod
    def prod(shape):
        """Compute shape product"""
        res = shape[0]
        for i in range(1, len(shape)):
            res = res * shape[i]
        return res

    def _cal_weight(self, ops):
        weight = 0
        for op in ops:
            weight += self.prod(op.output.shape) * \
                PrimLib.dtype_bytes(op.output.dtype)
        return weight

    def injective_analyze(self):
        """analyze injective case"""
        const_size = max((self.prod(op.output.shape) for op in self.dom_op))
        const_size = (const_size + self.MAX_NUM_THREADS -
                      1) // self.MAX_NUM_THREADS * self.MAX_NUM_THREADS

        total_weight = self._cal_weight(self.ops)
        total_block = (const_size + self.MAX_NUM_THREADS -
                       1) // self.MAX_NUM_THREADS
        need_block_split = const_size > self.MAX_BLOCK * self.MAX_NUM_THREADS
        if need_block_split:
            self.block_num = self.MAX_BLOCK
            waves = (total_block + self.MAX_BLOCK - 1) // self.MAX_BLOCK
            self.block_weight = total_weight // total_block * waves
        else:
            self.block_num = total_block
            self.block_weight = total_weight // self.block_num

    def reduce_analyze(self):
        """analyze reduce case"""
        thread_x, thread_y = 32, 32
        reduce_op = None
        for op in self.ops:
            if PrimLib.iter_type(op) == PrimLib.REDUCE:
                if reduce_op:
                    raise RuntimeError("Parallel fusion does not support multiple reduce op now.")
                reduce_op = op
        if not reduce_op:
            raise RuntimeError("Parallel fusion does not find a reduce op.")
        shape = reduce_op.inputs[0].shape
        reduce_axis = reduce_op.attrs['reduce_axis']
        total_space = self.prod(shape)
        red_space = shape[reduce_axis[0]]
        for i in range(1, len(reduce_axis)):
            red_space *= shape[reduce_axis[i]]
        dtype_size = PrimLib.dtype_bytes(reduce_op.output.dtype)

        weight = self._cal_weight(self.ops)  # reduce + injective
        block_x = (total_space // red_space + thread_y - 1) // thread_y
        block_w = (weight + block_x - 1) // block_x
        waves = (block_x + self.MAX_BLOCK - 1) // self.MAX_BLOCK
        self.block_num = min(self.MAX_BLOCK, block_x)
        all_reduce = 10  # 1 reduce init + 3 sync + 5 bin + 1 write
        self.block_weight = (block_w + all_reduce *
                             dtype_size * thread_x * thread_y) * waves

    def default_analyze(self):
        """analyze default case"""
        def _cal_default_space(op):
            space = self.prod(op.output.shape)
            for t in op.inputs:
                size = self.prod(t.shape)
                if size > space:
                    space = size
            return space
        space = max((_cal_default_space(op) for op in self.dom_op))

        # each sm least 4 wrap
        block = (space + (self.WRAP_SIZE * 4) - 1) // (self.WRAP_SIZE * 4)
        self.block_num = min(self.MAX_BLOCK, block)
        self.block_weight = self._cal_weight(self.ops) // self.block_num

    def analyze(self):
        """analyze ops"""
        def _ops_type(ops, dom_op):
            have_reduce = any(
                (PrimLib.iter_type(op) == PrimLib.REDUCE for op in ops))
            if have_reduce:
                return True
            return PrimLib.iter_type(dom_op[0])

        dom_type = _ops_type(self.ops, self.dom_op)
        if dom_type in (PrimLib.ELEMWISE, PrimLib.BROADCAST):
            self.injective_analyze()
        elif dom_type == PrimLib.REDUCE:
            self.reduce_analyze()
        else:
            self.default_analyze()

    def suitable_to_pipeline(self):
        """judge whether is suitable to be pipeline optimized"""
        # Reduce is not suitable
        def _contain_reduce(ops):
            for op in ops:
                # Reduce may make the tiling bad.
                if PrimLib.primtives.get(op.prim, None) == PrimLib.REDUCE:
                    return True
            return False

        suitable = True
        if _contain_reduce(self.ops):
            suitable = False
        return suitable

    @staticmethod
    def k_mean(data, class_n=2, exclude_id=()):
        """
        Find k clusters in which element is close to each other.

        Args:
            data (list): Elements' information.
            class_n (int): Number of clusters wanted to be analyzed, default is 2.
            exclude_id (tuple[int]): The list of excluded element's index, default is ().

        Returns:
            classes (list[list[int]]): The list of clusters. Each cluster is a list of indices.
        """
        def _cal_mean(classes):
            class_datas = list(list(data[cid] for cid in cls) for cls in classes)
            return list(sum(cls) / len(cls) if cls else float('inf') for cls in class_datas)

        def _cal_distance(a, b):
            return abs(a - b)

        def _check_different(old_classes, new_classes):
            for old_class, new_class in zip(old_classes, new_classes):
                if old_class != new_class:
                    return True
            return False

        if len(data) < class_n:
            return []
        classes = []
        for i, _ in enumerate(data):
            if i in exclude_id:
                continue
            if len(classes) >= class_n:
                break
            classes.append([i])
        changed = True
        while changed:
            new_classes = list([] for cls in classes)
            means = _cal_mean(classes)
            for idx, d in enumerate(data):
                if idx in exclude_id:
                    continue
                min_idx = -1
                min_dis = float('inf')
                for i, m in enumerate(means):
                    cur_dis = _cal_distance(m, d)
                    min_idx = i if min_dis > cur_dis else min_idx
                    min_dis = cur_dis if min_dis > cur_dis else min_dis
                new_classes[min_idx].append(idx)
            changed = _check_different(classes, new_classes)
            classes = new_classes
        return classes

    @staticmethod
    def pipeline_fusion_analyze(blocks, op_sizes, exclude_id):
        """analyze whether the segments can be pipeline optimized"""
        # op size first, block second.
        def _simple_factor(block, op_size):
            return block + 5 * op_size

        def _take_second(elem):
            return elem[1]

        simple_indicators = list(_simple_factor(b, s)
                                 for b, s in zip(blocks, op_sizes))
        # 2 classes, one heavy, the other light
        classes = ScheduleAnalyzer.k_mean(simple_indicators, 2, exclude_id)
        if not classes:
            return []
        means = list(sum([simple_indicators[idx] for idx in cls]) /
                     len(cls) if cls else float('inf') for cls in classes)

        # The target two clusters should be a heavy one and a light one.
        # The light one maybe suitable to run with pipeline optimized.
        classes_infos = list([cls, m] for cls, m in zip(classes, means))
        classes_infos.sort(key=_take_second)
        pipeline_target = None
        for ci in classes_infos:
            if ci:
                pipeline_target = ci
                break
        pipeline_gids, pipeline_mean = pipeline_target
        if pipeline_mean > _simple_factor(float(ScheduleAnalyzer.MAX_SM) / len(blocks),
                                          ScheduleAnalyzer.PIPELINE_OP_THREADHOLD):
            return []

        pipeline_blocks = []
        pipeline_weight = len(pipeline_gids)
        # Try to make two paralleled at least.
        if pipeline_weight > 3 and pipeline_weight > len(blocks) / 2:
            if len(pipeline_gids[:pipeline_weight // 2]) > 1:
                pipeline_blocks.append(pipeline_gids[:pipeline_weight // 2])
            if len(pipeline_gids[pipeline_weight // 2:]) > 1:
                pipeline_blocks.append(pipeline_gids[pipeline_weight // 2:])
        elif pipeline_weight > 1:
            pipeline_blocks.append(pipeline_gids)
        return pipeline_blocks

    @staticmethod
    def fusion_consult(blocks, op_sizes, exclude_gid):
        """get a recommendation for parallel fusion"""
        # Default is block fusion
        fusion_type = "block_fusion"
        type_info = None

        activate_pipeline_optimization = False  # Disable pipeline optimization for now.
        if activate_pipeline_optimization:
            pipeline_info = ScheduleAnalyzer.pipeline_fusion_analyze(
                blocks, op_sizes, exclude_gid)
            if pipeline_info:
                fusion_type = "block_pipeline_fusion"
                type_info = pipeline_info

        return fusion_type, type_info


def block_parallel_estimate(graphs):
    """estimate block parallel gain"""
    sum_block, max_weight, sum_weight, blocks, op_sizes, exclude_gid = 0, 0, 0, [], [], []
    for gid, g in enumerate(graphs):
        s = ScheduleAnalyzer(g)
        s.analyze()
        sum_block += s.block_num
        if s.block_weight > max_weight:
            max_weight = s.block_weight
        sum_weight += s.block_weight
        blocks.append(s.block_num)
        op_sizes.append(len(s.ops))
        if not s.suitable_to_pipeline():
            exclude_gid.append(gid)
    if sum_block > ScheduleAnalyzer.MAX_SM * 32:
        return ParalGain("none", sum_weight, 0, list(0 for _ in graphs), None)

    fusion_type, type_info = ScheduleAnalyzer.fusion_consult(blocks, op_sizes, tuple(exclude_gid))
    return ParalGain(fusion_type, max_weight, sum_weight - max_weight, blocks, type_info)


def parallel_estimate(graphs, target):
    """Estimate parallel gain"""
    if target == "aicore":
        fusion_type = "block_fusion"
        type_info = None
        fake_estimate = 1000
        fake_blocks = list(1 for g in graphs)
        return ParalGain(fusion_type, fake_estimate, fake_estimate, fake_blocks, type_info)
    return block_parallel_estimate(graphs)
