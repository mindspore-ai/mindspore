
# Copyright 2022 Huawei Technologies Co., Ltd
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

"""random_ops vmap impl."""
from ..operations.random_ops import UniformCandidateSampler
from .._vmap.vmap_base import vmap_rules_getters, _bdim_at_front, _vmap_clone_prim


@vmap_rules_getters.register(UniformCandidateSampler)
def get_uniform_candidate_sampler_vmap_rule(prim, axis_size):
    """VmapRule for `UniformCandidateSampler` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr("batch_rank", batch_rank)

    def vmap_rule(x_bdim):
        x, x_dim = x_bdim
        if x_dim is None:
            sampled_candidates, true_expected_count, sampled_expected_count = prim(x)
            return (sampled_candidates, None), (true_expected_count, None), (sampled_expected_count, None)

        x = _bdim_at_front(x, x_dim, axis_size)
        sampled_candidates, true_expected_count, sampled_expected_count = batch_prim(x)

        return (sampled_candidates, 0), (true_expected_count, 0), (sampled_expected_count, 0)

    return vmap_rule
