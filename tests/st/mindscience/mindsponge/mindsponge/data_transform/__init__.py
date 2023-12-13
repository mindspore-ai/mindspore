# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""Data"""


from .data import get_chi_atom_pos_indices
from .data_transform import one_hot, correct_msa_restypes,\
    randomly_replace_msa_with_unknown, fix_templates_aatype, pseudo_beta_fn, make_atom14_masks, \
    block_delete_msa_indices, sample_msa, make_masked_msa, \
    nearest_neighbor_clusters, summarize_clusters, crop_extra_msa, \
    make_msa_feat, random_crop_to_size, generate_random_sample

__all__ = ['one_hot', 'correct_msa_restypes', 'get_chi_atom_pos_indices', 'make_atom14_masks', \
    'randomly_replace_msa_with_unknown', 'fix_templates_aatype', 'pseudo_beta_fn', \
    'block_delete_msa_indices', 'sample_msa', 'make_masked_msa', \
    'nearest_neighbor_clusters', 'summarize_clusters', 'crop_extra_msa', \
    'make_msa_feat', 'random_crop_to_size', 'generate_random_sample']
