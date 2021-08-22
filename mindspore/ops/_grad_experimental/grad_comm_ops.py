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
# ============================================================================

"""Generate bprop for comm ops"""
from .._grad.grad_base import bprop_getters
from ..operations._inner_ops import NeighborExchange


@bprop_getters.register(NeighborExchange)
def get_bprop_neighborexchange(self):
    """Generate bprop for NeighborExchange."""
    group = self.group
    send_rank_ids = self.recv_rank_ids
    recv_rank_ids = self.send_rank_ids
    recv_shapes = self.send_shapes
    send_shapes = self.recv_shapes
    recv_type = self.recv_type
    neighborexchange_grad = NeighborExchange(send_rank_ids, recv_rank_ids, recv_shapes, send_shapes, recv_type, group)

    def bprop(x, out, dout):
        return (neighborexchange_grad(dout),)

    return bprop
