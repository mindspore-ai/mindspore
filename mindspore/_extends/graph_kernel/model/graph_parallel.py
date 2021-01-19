# Copyright 2021 Huawei Technologies Co., Ltd
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
from .model import PrimLib


class ParalGain:
    def __init__(self, fusion_type, bottleneck, gain, block_assign):
        self.fusion_type = fusion_type
        self.bottleneck = bottleneck
        self.gain = gain
        self.block_assign = block_assign


class ScheduleAnalyzer:
    """schedule analyzer"""
    WRAP_SIZE = 32
    MAX_SM = 80  # Volta
    MAX_NUM_THREADS = 1024
    MAX_BLOCK = 256

    def __init__(self, graph):
        self.graph = graph
        self.block_num = 0
        self.block_weight = 0
        _, outputs = graph.deduce_parameters()
        self.ops = graph.ops
        self.dom_op = [out.op for out in outputs]

    def prod(self, shape):
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
        const_size = max([self.prod(op.output.shape) for op in self.dom_op])
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
                    raise RuntimeError(
                        "Not support multiply reduce op in a graph now.")
                reduce_op = op
        if not reduce_op:
            raise RuntimeError("Wrong analyze for reduce!")
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
        space = max([_cal_default_space(op) for op in self.dom_op])

        # each sm least 4 wrap
        block = (space + (self.WRAP_SIZE * 4) - 1) // (self.WRAP_SIZE * 4)
        self.block_num = min(self.MAX_BLOCK, block)
        self.block_weight = self._cal_weight(self.ops) // self.block_num

    def analyze(self):
        """analyze ops"""
        def _ops_type(ops, dom_op):
            have_reduce = any(
                [PrimLib.iter_type(op) == PrimLib.REDUCE for op in ops])
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


def block_parallel_estimate(graphs):
    """estimate block parallel gain"""
    sum_block, max_weight, sum_weight, blocks = 0, 0, 0, []
    for g in graphs:
        s = ScheduleAnalyzer(g)
        s.analyze()
        sum_block += s.block_num
        if s.block_weight > max_weight:
            max_weight = s.block_weight
        sum_weight += s.block_weight
        blocks.append(s.block_num)
    if sum_block > ScheduleAnalyzer.MAX_SM * 32:
        return ParalGain("none", sum_weight, 0, [])
    return ParalGain("block_fusion", max_weight, sum_weight - max_weight, blocks)


def parallel_estimate(graphs):
    return block_parallel_estimate(graphs)
