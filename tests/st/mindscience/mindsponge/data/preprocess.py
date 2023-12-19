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
"""data process"""
import numpy as np

from tests.st.mindscience.mindsponge.mindsponge.data_transform import one_hot, correct_msa_restypes,\
    randomly_replace_msa_with_unknown, fix_templates_aatype, pseudo_beta_fn, make_atom14_masks, \
    block_delete_msa_indices, sample_msa, make_masked_msa, \
    nearest_neighbor_clusters, summarize_clusters, crop_extra_msa, \
    make_msa_feat, random_crop_to_size, generate_random_sample
from tests.st.mindscience.mindsponge.mindsponge.common.residue_constants import atom_type_num

NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'
NUM_SEQ = "length msa placeholder"
NUM_NOISE = 'num noise placeholder'
NUM_LATENT_DIM = "num latent placeholder"
_MSA_FEATURE_NAMES = ['msa', 'deletion_matrix', 'msa_mask', 'msa_row_mask', 'bert_mask', 'true_msa', 'msa_input']

FEATURES = {
    # Static features of a protein sequence
    "aatype": (np.float32, [NUM_RES, 21]),
    "between_segment_residues": (np.int64, [NUM_RES, 1]),
    "deletion_matrix": (np.float32, [NUM_SEQ, NUM_RES, 1]),
    "msa": (np.int64, [NUM_SEQ, NUM_RES, 1]),
    "num_alignments": (np.int64, [NUM_RES, 1]),
    "residue_index": (np.int64, [NUM_RES, 1]),
    "seq_length": (np.int64, [NUM_RES, 1]),
    "all_atom_positions": (np.float32, [NUM_RES, atom_type_num, 3]),
    "all_atom_mask": (np.int64, [NUM_RES, atom_type_num]),
    "resolution": (np.float32, [1]),
    "template_domain_names": (str, [NUM_TEMPLATES]),
    "template_sum_probs": (np.float32, [NUM_TEMPLATES, 1]),
    "template_aatype": (np.float32, [NUM_TEMPLATES, NUM_RES, 22]),
    "template_all_atom_positions": (np.float32, [NUM_TEMPLATES, NUM_RES, atom_type_num, 3]),
    "template_all_atom_masks": (np.float32, [NUM_TEMPLATES, NUM_RES, atom_type_num, 1]),
    "atom14_atom_exists": (np.float32, [NUM_RES, 14]),
    "atom14_gt_exists": (np.float32, [NUM_RES, 14]),
    "atom14_gt_positions": (np.float32, [NUM_RES, 14, 3]),
    "residx_atom14_to_atom37": (np.float32, [NUM_RES, 14]),
    "residx_atom37_to_atom14": (np.float32, [NUM_RES, 37]),
    "atom37_atom_exists": (np.float32, [NUM_RES, 37]),
    "atom14_alt_gt_positions": (np.float32, [NUM_RES, 14, 3]),
    "atom14_alt_gt_exists": (np.float32, [NUM_RES, 14]),
    "atom14_atom_is_ambiguous": (np.float32, [NUM_RES, 14]),
    "rigidgroups_gt_frames": (np.float32, [NUM_RES, 8, 12]),
    "rigidgroups_gt_exists": (np.float32, [NUM_RES, 8]),
    "rigidgroups_group_exists": (np.float32, [NUM_RES, 8]),
    "rigidgroups_group_is_ambiguous": (np.float32, [NUM_RES, 8]),
    "rigidgroups_alt_gt_frames": (np.float32, [NUM_RES, 8, 12]),
    "backbone_affine_tensor": (np.float32, [NUM_RES, 7]),
    "torsion_angles_sin_cos": (np.float32, [NUM_RES, 4, 2]),
    "torsion_angles_mask": (np.float32, [NUM_RES, 7]),
    "pseudo_beta": (np.float32, [NUM_RES, 3]),
    "pseudo_beta_mask": (np.float32, [NUM_RES]),
    "chi_mask": (np.float32, [NUM_RES, 4]),
    "backbone_affine_mask": (np.float32, [NUM_RES]),
}

feature_list = {
    'aatype': [NUM_RES],
    'all_atom_mask': [NUM_RES, None],
    'all_atom_positions': [NUM_RES, None, None],
    'alt_chi_angles': [NUM_RES, None],
    'atom14_alt_gt_exists': [NUM_RES, None],
    'atom14_alt_gt_positions': [NUM_RES, None, None],
    'atom14_atom_exists': [NUM_RES, None],
    'atom14_atom_is_ambiguous': [NUM_RES, None],
    'atom14_gt_exists': [NUM_RES, None],
    'atom14_gt_positions': [NUM_RES, None, None],
    'atom37_atom_exists': [NUM_RES, None],
    'backbone_affine_mask': [NUM_RES],
    'backbone_affine_tensor': [NUM_RES, None],
    'bert_mask': [NUM_MSA_SEQ, NUM_RES],
    'chi_angles': [NUM_RES, None],
    'chi_mask': [NUM_RES, None],
    'extra_deletion_value': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_has_deletion': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa_mask': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa_row_mask': [NUM_EXTRA_SEQ],
    'is_distillation': [],
    'msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
    'msa_mask': [NUM_MSA_SEQ, NUM_RES],
    'msa_row_mask': [NUM_MSA_SEQ],
    'pseudo_beta': [NUM_RES, None],
    'pseudo_beta_mask': [NUM_RES],
    'random_crop_to_size_seed': [None],
    'residue_index': [NUM_RES],
    'residx_atom14_to_atom37': [NUM_RES, None],
    'residx_atom37_to_atom14': [NUM_RES, None],
    'resolution': [],
    'rigidgroups_alt_gt_frames': [NUM_RES, None, None],
    'rigidgroups_group_exists': [NUM_RES, None],
    'rigidgroups_group_is_ambiguous': [NUM_RES, None],
    'rigidgroups_gt_exists': [NUM_RES, None],
    'rigidgroups_gt_frames': [NUM_RES, None, None],
    'seq_length': [],
    'seq_mask': [NUM_RES],
    'target_feat': [NUM_RES, None],
    'template_aatype': [NUM_TEMPLATES, NUM_RES],
    'template_all_atom_masks': [NUM_TEMPLATES, NUM_RES, None],
    'template_all_atom_positions': [
        NUM_TEMPLATES, NUM_RES, None, None],
    'template_backbone_affine_mask': [NUM_TEMPLATES, NUM_RES],
    'template_backbone_affine_tensor': [
        NUM_TEMPLATES, NUM_RES, None],
    'template_mask': [NUM_TEMPLATES],
    'template_pseudo_beta': [NUM_TEMPLATES, NUM_RES, None],
    'template_pseudo_beta_mask': [NUM_TEMPLATES, NUM_RES],
    'template_sum_probs': [NUM_TEMPLATES, None],
    'true_msa': [NUM_MSA_SEQ, NUM_RES],
    'torsion_angles_sin_cos': [NUM_RES, None, None],
    'msa_input': [NUM_MSA_SEQ, NUM_RES, 2],
    'query_input': [NUM_RES, 2],
    'additional_input': [NUM_RES, 4],
    'random_data': [NUM_NOISE, NUM_MSA_SEQ, NUM_RES, NUM_LATENT_DIM],
    'context_mask': [NUM_MSA_SEQ, 2]
}


def feature_shape(feature_name, num_residues, msa_length, num_templates, features=None):
    """Get the shape for the given feature name."""
    features = features or FEATURES
    if feature_name.endswith("_unnormalized"):
        feature_name = feature_name[:-13]
    unused_dtype, raw_sizes = features.get(feature_name, (None, None))
    replacements = {NUM_RES: num_residues,
                    NUM_SEQ: msa_length}

    if num_templates is not None:
        replacements[NUM_TEMPLATES] = num_templates

    sizes = [replacements.get(dimension, dimension) for dimension in raw_sizes]
    for dimension in sizes:
        if isinstance(dimension, str):
            raise ValueError("Could not parse %s (shape: %s) with values: %s" % (
                feature_name, raw_sizes, replacements))
    size_r = [int(x) for x in sizes]
    return size_r


def parse_reshape_logic(parsed_features, features, num_template, key=None):
    """Transforms parsed serial features to the correct shape."""
    # Find out what is the number of sequences and the number of alignments.
    num_residues = np.reshape(parsed_features['seq_length'].astype(np.int32), (-1,))[0]

    if "num_alignments" in parsed_features:
        num_msa = np.reshape(parsed_features["num_alignments"].astype(np.int32), (-1,))[0]
    else:
        num_msa = 0

    if key is not None and "key" in features:
        parsed_features["key"] = [key]  # Expand dims from () to (1,).

    # Reshape the arrays according to the sequence length and num alignments.
    for k, v in parsed_features.items():
        new_shape = feature_shape(
            feature_name=k,
            num_residues=num_residues,
            msa_length=num_msa,
            num_templates=num_template,
            features=features)
        new_shape_size = 1
        for dim in new_shape:
            new_shape_size *= dim

        if np.size(v) != new_shape_size:
            raise ValueError("the size of feature {} ({}) could not be reshaped into {}"
                             "".format(k, np.size(v), new_shape))

        if "template" not in k:
            # Make sure the feature we are reshaping is not empty.
            if np.size(v) <= 0:
                raise ValueError("The feature {} is not empty.".format(k))
        parsed_features[k] = np.reshape(v, new_shape)

    return parsed_features


def _make_features_metadata(feature_names):
    """Makes a feature name to type and shape mapping from a list of names."""
    # Make sure these features are always read.
    required_features = ["sequence", "domain_name", "template_domain_names"]
    feature_names = list(set(feature_names) - set(required_features))

    features_metadata = {name: FEATURES.get(name) for name in feature_names}
    return features_metadata


def np_to_array_dict(np_example, features):
    """Creates dict of arrays.

    Args:
      np_example: A dict of NumPy feature arrays.
      features: A list of strings of feature names to be returned in the dataset.

    Returns:
      A dictionary of features mapping feature names to features. Only the given
      features are returned, all other ones are filtered out.
    """
    features_metadata = _make_features_metadata(features)
    array_dict = {k: v for k, v in np_example.items() if k in features_metadata}
    if "template_domain_names" in np_example:
        num_template = len(np_example["template_domain_names"])
    else:
        num_template = 0

    # Ensures shapes are as expected. Needed for setting size of empty features
    # e.g. when no template hits were found.
    array_dict = parse_reshape_logic(array_dict, features_metadata, num_template)
    array_dict['template_mask'] = np.ones([num_template], np.float32)
    return array_dict


class Feature:
    """feature process"""

    def __init__(self, cfg, raw_feature=None, is_training=False, model_cfg=None, is_evogen=False):
        if raw_feature and isinstance(raw_feature, dict):
            self.ensemble_num = 0
            self.cfg = cfg
            self.model_cfg = model_cfg
            if 'deletion_matrix_int' in raw_feature:
                raw_feature['deletion_matrix'] = (raw_feature.pop('deletion_matrix_int').astype(np.float32))
            feature_names = cfg.common.unsupervised_features
            if cfg.common.use_templates:
                feature_names += cfg.common.template_features
            self.is_training = is_training
            self.is_evogen = is_evogen
            if self.is_training:
                feature_names += cfg.common.supervised_features
            raw_feature = np_to_array_dict(np_example=raw_feature, features=feature_names)

            for key in raw_feature:
                setattr(self, key, raw_feature[key])

    def non_ensemble(self, distillation=False, replace_proportion=0.0, use_templates=True):
        """non ensemble"""
        if self.is_evogen:
            msa, msa_input = correct_msa_restypes(self.msa, self.deletion_matrix, self.is_evogen)
            setattr(self, "msa", msa)
            setattr(self, "msa_input", msa_input.astype(np.float32))
        else:
            setattr(self, "msa", correct_msa_restypes(self.msa))
        setattr(self, "is_distillation", np.array(float(distillation), dtype=np.float32))
        # convert int64 to int32
        for k, v in vars(self).items():
            if k not in ("ensemble_num", "is_training", "is_evogen", "cfg", "model_cfg"):
                if v.dtype == np.int64:
                    setattr(self, k, v.astype(np.int32))
        aatype = np.argmax(self.aatype, axis=-1)
        setattr(self, "aatype", aatype.astype(np.int32))
        if self.is_evogen:
            query_input = np.concatenate((aatype[:, None], self.deletion_matrix[0]),
                                         axis=-1).astype(np.int32)
            setattr(self, "query_input", query_input.astype(np.float32))
        data = vars(self)
        for k in ['msa', 'num_alignments', 'seq_length', 'sequence', 'superfamily', 'deletion_matrix',
                  'resolution', 'between_segment_residues', 'residue_index', 'template_all_atom_masks']:
            if k in data:
                final_dim = data[k].shape[-1]
                if isinstance(final_dim, int) and final_dim == 1:
                    setattr(self, k, np.squeeze(data[k], axis=-1))
        # Remove fake sequence dimension
        for k in ['seq_length', 'num_alignments']:
            if k in data:
                setattr(self, k, data[k][0])

        msa, aatype = randomly_replace_msa_with_unknown(self.msa, self.aatype, replace_proportion)
        setattr(self, "msa", msa)
        setattr(self, "aatype", aatype)
        # seq_mask
        seq_mask = np.ones(self.aatype.shape, dtype=np.float32)
        setattr(self, "seq_mask", seq_mask)
        # msa_mask and msa_row_mask
        msa_mask = np.ones(self.msa.shape, dtype=np.float32)
        msa_row_mask = np.ones(self.msa.shape[0], dtype=np.float32)
        setattr(self, "msa_mask", msa_mask)
        setattr(self, "msa_row_mask", msa_row_mask)
        if 'hhblits_profile' not in data:
            # Compute the profile for every residue (over all MSA sequences).
            setattr(self, 'hhblits_profile', np.mean(one_hot(22, self.msa), axis=0))

        if use_templates:
            template_aatype = fix_templates_aatype(self.template_aatype)
            setattr(self, "template_aatype", template_aatype)
            template_pseudo_beta, template_pseudo_beta_mask = pseudo_beta_fn(self.template_aatype,
                                                                             self.template_all_atom_positions,
                                                                             self.template_all_atom_masks)
            setattr(self, "template_pseudo_beta", template_pseudo_beta)
            setattr(self, "template_pseudo_beta_mask", template_pseudo_beta_mask)

        atom14_atom_exists, residx_atom14_to_atom37, residx_atom37_to_atom14, atom37_atom_exists = \
            make_atom14_masks(self.aatype)
        setattr(self, "atom14_atom_exists", atom14_atom_exists)
        setattr(self, "residx_atom14_to_atom37", residx_atom14_to_atom37)
        setattr(self, "residx_atom37_to_atom14", residx_atom37_to_atom14)
        setattr(self, "atom37_atom_exists", atom37_atom_exists)

    def ensemble(self, inputs):
        """ensemble"""
        data = inputs[0]
        msa_fraction_per_block = inputs[1]
        randomize_num_blocks = inputs[2]
        num_blocks = inputs[3]
        keep_extra = inputs[4]
        max_msa_clusters = inputs[5]
        masked_msa = inputs[6]
        uniform_prob = inputs[7]
        profile_prob = inputs[8]
        same_prob = inputs[9]
        replace_fraction = inputs[10]
        msa_cluster_features = inputs[11]
        max_extra_msa = inputs[12]
        crop_size = inputs[13]
        max_templates = inputs[14]
        subsample_templates = inputs[15]
        fixed_size = inputs[16]
        seed = inputs[17]
        random_recycle = inputs[18]
        self.ensemble_num += 1
        if self.is_training:
            keep_indices = block_delete_msa_indices(data["msa"], msa_fraction_per_block, randomize_num_blocks,
                                                    num_blocks)
            for k in _MSA_FEATURE_NAMES:
                if k in data:
                    data[k] = data[k][keep_indices]
        # exist numpy random op
        is_sel, not_sel_seq, sel_seq = sample_msa(data["msa"], max_msa_clusters)
        for k in _MSA_FEATURE_NAMES:
            if k in data:
                if keep_extra and not is_sel:
                    new_shape = list(data[k].shape)
                    new_shape[0] = 1
                    data['extra_' + k] = np.zeros(new_shape)
                elif keep_extra and is_sel:
                    data['extra_' + k] = data[k][not_sel_seq]
                if k == 'msa':
                    data['extra_msa'] = data['extra_msa'].astype(np.int32)
                data[k] = data[k][sel_seq]
        if masked_msa:
            inputs = (data["msa"], data["hhblits_profile"], uniform_prob, profile_prob, same_prob, replace_fraction)
            data["bert_mask"], data["true_msa"], data["msa"] = make_masked_msa(inputs)
        if msa_cluster_features:
            data["extra_cluster_assignment"] = nearest_neighbor_clusters(data["msa_mask"], data["msa"],
                                                                         data["extra_msa_mask"], data["extra_msa"])
            inputs = (data["msa"], data["msa_mask"], data["extra_cluster_assignment"], data["extra_msa_mask"], \
                      data["extra_msa"], data["extra_deletion_matrix"], data["deletion_matrix"])
            data["cluster_profile"], data["cluster_deletion_mean"] = summarize_clusters(inputs)

        if max_extra_msa:
            select_indices = crop_extra_msa(data["extra_msa"], max_extra_msa)
            if select_indices:
                for k in _MSA_FEATURE_NAMES:
                    if 'extra_' + k in data:
                        data['extra_' + k] = data['extra_' + k][select_indices]
        else:
            for k in _MSA_FEATURE_NAMES:
                if 'extra_' + k in data:
                    del data['extra_' + k]
        inputs = (data["between_segment_residues"], data["aatype"], data["msa"], data["deletion_matrix"],
                  data["cluster_deletion_mean"], data["cluster_profile"], data["extra_deletion_matrix"])
        data["extra_has_deletion"], data["extra_deletion_value"], data["msa_feat"], \
        data["target_feat"] = make_msa_feat(inputs)

        if fixed_size:
            data = {k: v for k, v in data.items() if k in feature_list}

            inputs = (data["seq_length"], data["template_mask"], crop_size, max_templates,
                      subsample_templates, seed, random_recycle)
            num_res_crop_size, num_templates_crop_size_int, num_res_crop_start, num_res_crop_size_int, \
            templates_crop_start, templates_select_indices = random_crop_to_size(inputs)
            for k, v in data.items():
                if k not in feature_list or ('template' not in k and NUM_RES not in feature_list.get(k)):
                    continue

                # randomly permute the templates before cropping them.
                if k.startswith('template') and subsample_templates:
                    v = v[templates_select_indices]

                crop_sizes = []
                crop_starts = []
                for i, (dim_size, dim) in enumerate(zip(feature_list.get(k), v.shape)):
                    is_num_res = (dim_size == NUM_RES)
                    if i == 0 and k.startswith('template'):
                        crop_size_ = num_templates_crop_size_int
                        crop_start = templates_crop_start
                    else:
                        crop_start = num_res_crop_start if is_num_res else 0
                        crop_size_ = (num_res_crop_size_int if is_num_res else (-1 if dim is None else dim))
                    crop_sizes.append(crop_size_)
                    crop_starts.append(crop_start)
                if len(v.shape) == 1:
                    data[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0]]
                elif len(v.shape) == 2:
                    data[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                                crop_starts[1]:crop_starts[1] + crop_sizes[1]]
                elif len(v.shape) == 3:
                    data[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                                crop_starts[1]:crop_starts[1] + crop_sizes[1],
                                crop_starts[2]:crop_starts[2] + crop_sizes[2]]
                else:
                    data[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                                crop_starts[1]:crop_starts[1] + crop_sizes[1],
                                crop_starts[2]:crop_starts[2] + crop_sizes[2],
                                crop_starts[3]:crop_starts[3] + crop_sizes[3]]

            data["seq_length"] = num_res_crop_size

            pad_size_map = {
                NUM_RES: crop_size,
                NUM_MSA_SEQ: max_msa_clusters,
                NUM_EXTRA_SEQ: max_extra_msa,
                NUM_TEMPLATES: max_templates,
            }

            for k, v in data.items():
                if k == 'extra_cluster_assignment':
                    continue
                shape = list(v.shape)
                schema = feature_list.get(k)
                pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]
                padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
                if padding:
                    data[k] = np.pad(v, padding)
                    data[k].reshape(pad_size)
        else:
            for k, v in data.items():
                if k.startswith('template_'):
                    data[k] = v[:max_templates]
        if self.is_evogen:
            data["random_data"], data["context_mask"] = generate_random_sample(self.cfg, self.model_cfg)
            data["context_mask"] = data["context_mask"].astype(np.float32)
        return data

    def process_res(self, features, res, dtype):
        """process result"""
        arrays, prev_pos, prev_msa_first_row, prev_pair = res
        if self.is_evogen:
            evogen_keys = ["target_feat", "seq_mask", "aatype", "residx_atom37_to_atom14", "atom37_atom_exists",
                           "residue_index", "msa_mask", "msa_input", "query_input", "additional_input", "random_data",
                           "context_mask"]
            arrays = [features[key] for key in evogen_keys]
            arrays = [array.astype(dtype) if array.dtype == "float64" else array for array in arrays]
            arrays = [array.astype(dtype) if array.dtype == "float32" else array for array in arrays]
            res = [arrays, prev_pos, prev_msa_first_row, prev_pair]
            return res
        if self.is_training:
            label_keys = ["pseudo_beta", "pseudo_beta_mask", "all_atom_mask",
                          "true_msa", "bert_mask", "residue_index", "seq_mask",
                          "atom37_atom_exists", "aatype", "residx_atom14_to_atom37",
                          "atom14_atom_exists", "backbone_affine_tensor", "backbone_affine_mask",
                          "atom14_gt_positions", "atom14_alt_gt_positions",
                          "atom14_atom_is_ambiguous", "atom14_gt_exists", "atom14_alt_gt_exists",
                          "all_atom_positions", "rigidgroups_gt_frames", "rigidgroups_gt_exists",
                          "rigidgroups_alt_gt_frames", "torsion_angles_sin_cos", "chi_mask"]
            label_arrays = [features[key] for key in label_keys]
            for i, _ in enumerate(label_arrays):
                if i not in (3, 4):
                    label_arrays[i] = label_arrays[i][0]
            label_arrays = [array.astype(dtype) if array.dtype == "float64" else array for array in label_arrays]
            label_arrays = [array.astype(dtype) if array.dtype == "float32" else array for array in label_arrays]
            res = [arrays, prev_pos, prev_msa_first_row, prev_pair, label_arrays]
            return res
        return res

    def pipeline(self, cfg, mixed_precision=True, seed=0):
        """feature process pipeline"""
        self.non_ensemble(cfg.common.distillation, cfg.common.replace_proportion, cfg.common.use_templates)
        non_ensemble_data = vars(self).copy()
        max_msa_clusters = cfg.eval.max_msa_clusters
        if cfg.common.reduce_msa_clusters_by_max_templates and not self.is_evogen:
            max_msa_clusters = cfg.eval.max_msa_clusters - cfg.eval.max_templates
        random_recycle = cfg.common.random_recycle
        non_ensemble_data_copy = non_ensemble_data.copy()
        inputs = (non_ensemble_data_copy,
                  cfg.block_deletion.msa_fraction_per_block,
                  cfg.block_deletion.randomize_num_blocks,
                  cfg.block_deletion.num_blocks,
                  cfg.eval.keep_extra,
                  max_msa_clusters,
                  cfg.common.masked_msa.use_masked_msa,
                  cfg.common.masked_msa.uniform_prob,
                  cfg.common.masked_msa.profile_prob,
                  cfg.common.masked_msa.same_prob,
                  cfg.eval.masked_msa_replace_fraction,
                  cfg.common.msa_cluster_features,
                  cfg.common.max_extra_msa,
                  cfg.eval.crop_size,
                  cfg.eval.max_templates,
                  cfg.eval.subsample_templates,
                  cfg.eval.fixed_size, seed, random_recycle)
        protein = self.ensemble(inputs)
        num_ensemble = cfg.eval.num_ensemble
        num_recycle = cfg.common.num_recycle
        if cfg.common.resample_msa_in_recycling:
            num_ensemble *= num_recycle
        result_array = {x: () for x in protein.keys()}
        if num_ensemble > 1:
            for _ in range(num_ensemble):
                non_ensemble_data_copy = non_ensemble_data.copy()
                inputs = (non_ensemble_data_copy,
                          cfg.block_deletion.msa_fraction_per_block,
                          cfg.block_deletion.randomize_num_blocks,
                          cfg.block_deletion.num_blocks, cfg.eval.keep_extra,
                          max_msa_clusters, cfg.common.masked_msa.use_masked_msa,
                          cfg.common.masked_msa.uniform_prob, cfg.common.masked_msa.profile_prob,
                          cfg.common.masked_msa.same_prob, cfg.eval.masked_msa_replace_fraction,
                          cfg.common.msa_cluster_features, cfg.common.max_extra_msa,
                          cfg.eval.crop_size, cfg.eval.max_templates, cfg.eval.subsample_templates,
                          cfg.eval.fixed_size, seed, random_recycle)
                data_t = self.ensemble(inputs)
                for key in protein.keys():
                    result_array[key] += (data_t[key][None],)
            for key in protein.keys():
                result_array[key] = np.concatenate(result_array[key], axis=0)
        else:
            result_array = {key: protein[key][None] for key in protein.keys()}
        features = {k: v for k, v in result_array.items() if v.dtype != 'O'}
        extra_msa_length = cfg.common.max_extra_msa
        for key in ["extra_msa", "extra_has_deletion", "extra_deletion_value", "extra_msa_mask"]:
            features[key] = features[key][:, :extra_msa_length]
        input_keys = ['target_feat', 'msa_feat', 'msa_mask', 'seq_mask', 'aatype', 'template_aatype',
                      'template_all_atom_masks', 'template_all_atom_positions', 'template_mask',
                      'template_pseudo_beta_mask', 'template_pseudo_beta',
                      'extra_msa', 'extra_has_deletion', 'extra_deletion_value', 'extra_msa_mask',
                      'residx_atom37_to_atom14', 'atom37_atom_exists', 'residue_index']

        dtype = np.float32
        if mixed_precision:
            dtype = np.float16
        arrays = [features[key] for key in input_keys]
        arrays = [array.astype(dtype) if array.dtype == "float64" else array for array in arrays]
        arrays = [array.astype(dtype) if array.dtype == "float32" else array for array in arrays]
        prev_pos = np.zeros([cfg.eval.crop_size, 37, 3]).astype(dtype)
        prev_msa_first_row = np.zeros([cfg.eval.crop_size, 256]).astype(dtype)
        prev_pair = np.zeros([cfg.eval.crop_size, cfg.eval.crop_size, 128]).astype(dtype)
        res = [arrays, prev_pos, prev_msa_first_row, prev_pair]
        res = self.process_res(features, res, dtype)
        return res
