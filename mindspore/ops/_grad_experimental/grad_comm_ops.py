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
from ..operations._inner_ops import AllToAllv


@bprop_getters.register(AllToAllv)
def get_bprop_alltoallv(self):
    """Generate bprop for AllToAllv."""
    group = self.group
    send_rank_ids = self.recv_rank_ids
    recv_rank_ids = self.send_rank_ids
    recv_shapes = self.recv_shapes_backward
    recv_type = self.recv_type
    alltoallv_grad = AllToAllv(send_rank_ids, recv_rank_ids, recv_shapes, recv_shapes, recv_type, group)

    def bprop(x, out, dout):
        return (alltoallv_grad(dout),)
    return bprop
