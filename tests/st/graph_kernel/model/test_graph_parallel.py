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
# ==========================================================================
"""test graph parallel case"""
import model

def injective_graph(shape):
    gb = model.GraphBuilder()
    with gb.graph_scope('injective') as _:
        a1 = gb.tensor(shape, 'float32')
        a2 = gb.emit('Abs', a1)
        a3 = gb.emit('Abs', a2)
        gb.emit('Abs', a3)
    return gb.get()[0]

def reduce_graph(shape, reduce_axis):
    gb = model.GraphBuilder()
    with gb.graph_scope('reduce') as _:
        a1 = gb.tensor(shape, 'float32')
        a2 = gb.emit('Abs', a1)
        a3 = gb.emit('Abs', a2)
        gb.emit('ReduceSum', a3, 'C', attrs={'reduce_axis': reduce_axis})
    return gb.get()[0]

def block_fusion(graphs):
    gain = model.parallel_estimate(graphs)
    print("fusion = {}, bottleneck = {}, gain = {}".format(gain.fusion_type, gain.bottleneck, gain.gain))
    return gain.fusion_type == "block_fusion" and gain.gain > 0

if __name__ == "__main__":
    assert block_fusion([injective_graph([40, 1024]), injective_graph([40, 1024])])
    assert block_fusion([reduce_graph([1024, 1024], [1]), injective_graph([24, 1024])])
    assert not block_fusion([reduce_graph([1024, 1024], [1]), injective_graph([50, 1024])])
    assert not block_fusion([reduce_graph([1024, 1024], [0, 1]), injective_graph([1024, 1024])])
