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
from __future__ import absolute_import

from mindspore.ops.operations.random_ops import UniformCandidateSampler, RandomShuffle, Multinomial, \
    RandomChoiceWithMask
from mindspore.ops.function import _VmapGeneralRule
from mindspore.ops._vmap.vmap_base import vmap_rules_getters, _bdim_at_front, _vmap_clone_prim, \
    vmap_general_preprocess, _raise_value_error


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


@vmap_rules_getters.register(RandomShuffle)
def get_random_shuffle_vmap_rule(prim, axis_size):
    """VmapRule for `RandomShuffle` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr("batch_rank", batch_rank)

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        out = batch_prim(x)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(Multinomial)
def get_multinomial_vmap_rule(prim, axis_size):
    """VmapRule for `Multinomial` operation."""
    prim_name = prim.name
    prim_vmap = _VmapGeneralRule(prim, axis_size)

    def vmap_rule(x_bdim, num_samples_bdim):
        is_all_none, result = vmap_general_preprocess(
            prim, x_bdim, num_samples_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        num_samples, num_samples_dim = num_samples_bdim
        if len(x.shape) > 2:
            out = prim_vmap(x_bdim, num_samples_bdim)
            return out
        if num_samples_dim is not None:
            _raise_value_error("The source axis of args in {} must be None, "
                               "but got {}.".format(prim_name, num_samples_dim))
        x = _bdim_at_front(x, x_dim, axis_size)
        out = prim(x, num_samples)
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(RandomChoiceWithMask)
def get_random_choice_with_mask(prim, axis_size):
    """VmapRule for 'RandomChoiceWithMask' operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        x_data, x_dim = x_bdim
        x = _bdim_at_front(x_data, x_dim, axis_size)
        index, mask = batch_prim(x)
        return (index, 0), (mask, 0)

    return vmap_rule
