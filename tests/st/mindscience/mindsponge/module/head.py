# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
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
"""structure module"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.ops import functional as F
from tests.st.mindscience.mindsponge.mindsponge.cell.initializer import lecun_init


class PredictedLDDTHead(nn.Cell):
    """Head to predict the per-residue LDDT to be used as a confidence measure."""

    def __init__(self, config, seq_channel):
        super().__init__()
        self.config = config
        self.input_layer_norm = nn.LayerNorm([seq_channel,], epsilon=1e-5)
        self.act_0 = nn.Dense(seq_channel, self.config.num_channels,
                              weight_init=lecun_init(seq_channel, initializer_name='relu')
                              ).to_float(mstype.float16)
        self.act_1 = nn.Dense(self.config.num_channels, self.config.num_channels,
                              weight_init=lecun_init(self.config.num_channels,
                                                     initializer_name='relu')
                              ).to_float(mstype.float16)
        self.logits = nn.Dense(self.config.num_channels, self.config.num_bins, weight_init='zeros'
                               ).to_float(mstype.float16)
        self.relu = nn.ReLU()

    def construct(self, rp_structure_module):
        """Builds ExperimentallyResolvedHead module."""
        act = rp_structure_module
        act = self.input_layer_norm(act.astype(mstype.float32))
        act = self.act_0(act)
        act = self.relu(act.astype(mstype.float32))
        act = self.act_1(act)
        act = self.relu(act.astype(mstype.float32))
        logits = self.logits(act)
        return logits


class DistogramHead(nn.Cell):
    """Head to predict a distogram.

    Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
    """

    def __init__(self, config, pair_dim):
        super().__init__()
        self.config = config
        self.half_logits = nn.Dense(pair_dim, self.config.num_bins, weight_init='zeros')
        self.first_break = self.config.first_break
        self.last_break = self.config.last_break
        self.num_bins = self.config.num_bins

    def construct(self, pair):
        """Builds DistogramHead module.

        Arguments:
          representations: Dictionary of representations, must contain:
            * 'pair': pair representation, shape [N_res, N_res, c_z].

        Returns:
          Dictionary containing:
            * logits: logits for distogram, shape [N_res, N_res, N_bins].
            * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
        """
        half_logits = self.half_logits(pair)

        logits = half_logits + mnp.swapaxes(half_logits, -2, -3)
        breaks = mnp.linspace(self.first_break, self.last_break, self.num_bins - 1)

        return logits, breaks


class ExperimentallyResolvedHead(nn.Cell):
    """Predicts if an atom is experimentally resolved in a high-res structure.

    Only trained on high-resolution X-ray crystals & cryo-EM.
    Jumper et al. (2021) Suppl. Sec. 1.9.10 '"Experimentally resolved" prediction'
    """

    def __init__(self, seq_channel):
        super().__init__()
        self.logits = nn.Dense(seq_channel, 37, weight_init='zeros')

    def construct(self, single):
        """Builds ExperimentallyResolvedHead module.

        Arguments:
          representations: Dictionary of representations, must contain:
            * 'single': Single representation, shape [N_res, c_s].

        Returns:
          Dictionary containing:
            * 'logits': logits of shape [N_res, 37],
                log probability that an atom is resolved in atom37 representation,
                can be converted to probability by applying sigmoid.
        """
        logits = self.logits(single)
        return logits


class MaskedMsaHead(nn.Cell):
    """Head to predict MSA at the masked locations.

    The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
    version of the full MSA, based on a linear projection of
    the MSA representation.
    Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
    """

    def __init__(self, config, msa_channel):
        super().__init__()
        self.config = config
        self.logits = nn.Dense(msa_channel, self.config.num_output, weight_init='zeros')

    def construct(self, msa):
        """Builds MaskedMsaHead module.

        Arguments:
          representations: Dictionary of representations, must contain:
            * 'msa': MSA representation, shape [N_seq, N_res, c_m].

        Returns:
          Dictionary containing:
            * 'logits': logits of shape [N_seq, N_res, N_aatype] with
                (unnormalized) log probabilies of predicted aatype at position.
        """
        # del batch
        logits = self.logits(msa)
        return logits


class PredictedAlignedErrorHead(nn.Cell):
    """Head to predict the distance errors in the backbone alignment frames.

    Can be used to compute predicted TM-Score.
    Jumper et al. (2021) Suppl. Sec. 1.9.7 "TM-score prediction"
    """

    def __init__(self, config, pair_dim):
        super().__init__()
        self.config = config
        self.num_bins = self.config.num_bins
        self.max_error_bin = self.config.max_error_bin
        self.logits = nn.Dense(pair_dim, self.num_bins, weight_init='zeros')

    def construct(self, pair):
        """Builds PredictedAlignedErrorHead module.

        Arguments:
            * 'pair': pair representation, shape [N_res, N_res, c_z].

        Returns:
            * logits: logits for aligned error, shape [N_res, N_res, N_bins].
            * breaks: array containing bin breaks, shape [N_bins - 1].
        """
        logits = self.logits(pair)
        breaks = mnp.linspace(0, self.max_error_bin, self.num_bins - 1)
        return logits, breaks


class EstogramHead(nn.Cell):
    """Head to predict estogram."""

    def __init__(self, first_break, last_break, num_bins):
        super().__init__()
        self.first_break = first_break
        self.last_break = last_break
        self.num_bins = num_bins

        self.breaks = np.linspace(self.first_break, self.last_break, self.num_bins)
        self.width = self.breaks[1] - self.breaks[0]

        self.centers = Tensor(self.breaks + 0.5 * self.width, mstype.float32)

        self.softmax = nn.Softmax(-1)
        self.zero = Tensor([0.])

    def compute_estogram(self, distogram_logits, decoy_distance_mat):
        """compute estogram matrix.
        Arguments:
            distogram_logits: [N_res, N_res, N_bins].
            decoy_distance_mat: [N_res, N_res]
        Returns:
            estogram: shape [N_res, N_res, N_bins].
            esto_centers: shape [N_res, N_res, N_bins].
        """
        square_centers = mnp.reshape(self.centers, (1, 1, -1))
        estogram = self.softmax(distogram_logits)
        esto_centers = square_centers - mnp.expand_dims(decoy_distance_mat, -1)
        return estogram, esto_centers

    def construct(self, distogram_logits, pseudo_beta, pseudo_beta_mask, cutoff=15.):
        """construct"""
        positions = pseudo_beta
        pad_mask = mnp.expand_dims(pseudo_beta_mask, 1)
        pad_mask_2d = pad_mask * mnp.transpose(pad_mask, (1, 0))
        pad_mask_2d *= (1. - mnp.eye(pad_mask_2d.shape[1]))

        dist_xyz = mnp.square(mnp.expand_dims(positions, axis=1) - \
                              mnp.expand_dims(positions, axis=0))
        dmat_decoy = mnp.sqrt(1e-10 + mnp.sum(dist_xyz.astype(mstype.float32), -1))

        estogram, esto_centers = self.compute_estogram(distogram_logits, dmat_decoy)
        pair_errors = mnp.sum(estogram * esto_centers, -1)

        p1 = self._integrate(distogram_logits, mnp.abs(esto_centers) < 0.5).astype(mnp.float32)
        p2 = self._integrate(distogram_logits, mnp.abs(esto_centers) < 1.0).astype(mnp.float32)
        p3 = self._integrate(distogram_logits, mnp.abs(esto_centers) < 2.0).astype(mnp.float32)
        p4 = self._integrate(distogram_logits, mnp.abs(esto_centers) < 4.0).astype(mnp.float32)

        p0 = self._integrate(distogram_logits, self.centers < cutoff).astype(mnp.float32)
        pred_mask2d = p0 * pad_mask_2d

        norm = mnp.sum(pred_mask2d, -1) + 1e-6
        p1 = mnp.sum(p1 * pred_mask2d, -1)
        p2 = mnp.sum(p2 * pred_mask2d, -1)
        p3 = mnp.sum(p3 * pred_mask2d, -1)
        p4 = mnp.sum(p4 * pred_mask2d, -1)

        plddt = 0.25 * (p1 + p2 + p3 + p4) / norm

        return plddt, pred_mask2d, pair_errors

    def _integrate(self, distogram_logits, integrate_masks):
        """compute estogram matrix.
        Arguments:
            distogram_logits: [N_res, N_res, N_bins].
            integrate_masks: [N_res, N_res, N_bins]
        Returns:
            v: shape [N_res, N_res].
        """
        probs = self.softmax(distogram_logits)
        integrate_masks = F.cast(integrate_masks, mnp.float32)
        v = mnp.sum(probs * integrate_masks, -1)
        return v
