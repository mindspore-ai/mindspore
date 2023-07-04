# Copyright 2023 Huawei Technologies Co., Ltd
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
"""The backward tik ops of flash attention"""
from tbe import tik

from mindspore.ops._op_impl._custom_op.flash_attention.tiling_strategy.strategy import TilingStrategy
from mindspore.ops._op_impl._custom_op.flash_attention.attention import FlashAttention

from mindspore.ops._op_impl._custom_op.flash_attention.constants import FP16
from mindspore.ops._op_impl._custom_op.flash_attention.constants import FP32
from mindspore.ops._op_impl._custom_op.flash_attention.constants import GM
from mindspore.ops._op_impl._custom_op.flash_attention.constants import L1
from mindspore.ops._op_impl._custom_op.flash_attention.constants import UB


class FlashAttentionBwd(FlashAttention):
    """The implementation of FlashAttention backward
    This function contains the flash attention backward implementation used in flash attention (see paper)
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`
    """

    def __init__(self, query, key, value, output, dO, rowsum, rowmax, dim_mask, attn_mask, dropout_mask, alibi_mask,
                 prev_block_num,
                 next_block_num,
                 high_precision,
                 kernel_name,
                 tiling_stgy: TilingStrategy,
                 disable_debug):
        super().__init__(query, key, value, dim_mask, attn_mask, dropout_mask, alibi_mask, kernel_name,
                         tiling_stgy, prev_block_num, next_block_num, high_precision, disable_debug)

        if isinstance(query, dict):
            self.dO_shape = dO["shape"]  # [B, Nq, d]
        else:
            self.dO_shape = dO.shape

        self.dV_shape = self.v_shape
        self.dQ_shape = self.q_shape
        self.dK_shape = self.k_shape
        self.dQ_gm = None
        self.dK_gm = None
        self.dV_gm = None
        self.O_gm = None
        self.dO_gm = None
        self.l_gm = None
        self.m_gm = None

    def define_outputs(self):
        """define output gm tensors"""
        self.dQ_gm = self.tik_instance.Tensor(FP32, self.dQ_shape, name="dQ_gm", scope=GM, is_atomic_add=True)
        self.dK_gm = self.tik_instance.Tensor(FP32, self.dK_shape, name="dK_gm", scope=GM, is_atomic_add=True)
        self.dV_gm = self.tik_instance.Tensor(FP32, self.dV_shape, name="dV_gm", scope=GM, is_atomic_add=True)

    def define_custom_inputs(self):
        """define input gm tensors"""
        self.O_gm = self.tik_instance.Tensor(FP16, self.O_shape, name="O_gm", scope=GM)
        self.dO_gm = self.tik_instance.Tensor(FP16, self.dO_shape, name="dO_gm", scope=GM)
        self.l_gm = self.tik_instance.Tensor(self.precision_type, self.l_shape, name="l_gm", scope=GM)
        self.m_gm = self.tik_instance.Tensor(FP16, self.m_shape, name="m_gm", scope=GM)

    def collect_inputs(self):
        """collect all input gm tensors into input_gm_list,
        the input list should keep order with the para order in Primitive and init
        """
        input_gm_list = [
            self.Q_gm, self.K_gm, self.V_gm, self.O_gm, self.dO_gm, self.l_gm,
            self.m_gm, self.dim_mask_gm
        ]
        if self.has_attn_mask:
            input_gm_list.append(self.att_mask_gm)
        if self.has_drop_mask:
            input_gm_list.append(self.drop_mask_gm)
        if self.has_alibi_mask:
            input_gm_list.append(self.alibi_mask_gm)
        return input_gm_list

    def compute_Pij(self, Qi_l1_K1MK0_ed, KjT_l1_K1NK0_ed, m, k, n, lm_gm_offset, attn_mask_gm_offset,
                    dropout_mask_gm_offset, alibi_mask_gm_offset):
        """Refer to Algorithm 4 line11-14 in FlashAttention implement Pij computation"""
        m_aligned = self.tik_ops_utils.up_align_to_K0(m)
        n_aligned = self.tik_ops_utils.up_align_to_K0(n)
        # Sij <- Qi * KjT
        Sij_ub = self.tik_ops_utils.matmul_compute(Qi_l1_K1MK0_ed, KjT_l1_K1NK0_ed, m, k, n)
        if self.has_drop_mask:
            Pij_drop_ed_ub = self.tik_instance.Tensor(FP16, (m_aligned, n_aligned), name="Pij_drop_ed_ub", scope=UB)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            if self.has_alibi_mask:
                self.do_alibi_mask(Sij_ub, alibi_mask_gm_offset, m_aligned, n_aligned)
            if self.has_attn_mask:
                self.do_att_mask(Sij_ub, attn_mask_gm_offset, m, n, m_aligned, n_aligned)

            # move li (ith block of l_gm) and mi (ith block of m_gm) from gm to ub
            li_ub = self.tik_instance.Tensor(self.precision_type, (m_aligned,), name="li_ub", scope=UB)
            mi_ub = self.tik_instance.Tensor(FP16, (m_aligned,), name="mi_ub", scope=UB)
            self.tik_ops_utils.move_vector_from_gm_to_ub(li_ub, self.l_gm, lm_gm_offset, m)
            self.tik_ops_utils.move_vector_from_gm_to_ub(mi_ub, self.m_gm, lm_gm_offset, m)
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                broadcast_mi_ub = self.tik_ops_utils.broadcast(mi_ub, (m_aligned, n_aligned))
                self.tik_instance.h_sub(Sij_ub, Sij_ub, broadcast_mi_ub)
            li_rec_ub = self.tik_ops_utils.calc_vec_rec(li_ub, m)
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                if self.high_precision:
                    Sij_ub_fp32 = self.tik_instance.Tensor(FP32, (m_aligned, n_aligned), name="Sij_ub_fp32", scope=UB)
                    self.tik_instance.h_cast(Sij_ub_fp32, Sij_ub, "none")
                    self.tik_instance.h_exp(Sij_ub_fp32, Sij_ub_fp32)
                    with self.tik_instance.for_range(0, m) as idx:
                        cur_row_sum_rec = self.tik_instance.Scalar(FP32, name="cur_row_sum_rec",
                                                                   init_value=li_rec_ub[idx])
                        self.tik_instance.h_mul(Sij_ub_fp32[idx, :], Sij_ub_fp32[idx, :], cur_row_sum_rec)
                    self.tik_instance.h_cast(Sij_ub, Sij_ub_fp32, "none")
                else:
                    self.tik_instance.h_exp(Sij_ub, Sij_ub)
                    broadcast_li_rec_ub = self.tik_ops_utils.broadcast(li_rec_ub, (m_aligned, n_aligned))
                    self.tik_instance.h_mul(Sij_ub, Sij_ub, broadcast_li_rec_ub)

            # dropout_mask
            if self.has_drop_mask:
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    dropout_mask_ub = self.tik_instance.Tensor(FP16, (m_aligned, n_aligned),
                                                               scope=UB, name="drop_mask_ub")
                    self.tik_instance.data_move(dropout_mask_ub, self.drop_mask_gm[dropout_mask_gm_offset], 0,
                                                m_aligned, n_aligned // 16, (self.N - n_aligned) // 16, 0)
                    self.tik_instance.h_mul(Pij_drop_ed_ub, Sij_ub, dropout_mask_ub)
            else:
                Pij_drop_ed_ub = Sij_ub

        return Sij_ub, Pij_drop_ed_ub

    def compute_Di(self, Di_ub, dOi_ub, qo_gm_offset, q_blk_height):
        """Refer to Algorithm 4 line19 in FlashAttention implement Di computation"""
        q_blk_height_aligned = self.tik_ops_utils.up_align_to_K0(q_blk_height)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Oi_ub = self.tik_instance.Tensor(FP16, (q_blk_height_aligned, self.d), name="Oi_ub", scope=UB)
            self.cont_data_mv_1_bust(dst=Oi_ub, src=self.O_gm[qo_gm_offset],
                                     burst=q_blk_height * self.d // 16)
            self.tik_instance.h_mul(Oi_ub, dOi_ub, Oi_ub)
            dOi_Oi_l1 = self.tik_instance.Tensor(FP16, (q_blk_height_aligned, self.d), name="dOi_Oi_l1", scope=L1)
            dOi_Oi_l1_K1MK0_ed = self.tik_ops_utils.MK_TO_K1MK0(Oi_ub, workspace_tensor=dOi_Oi_l1)
            self.tik_ops_utils.row_sum_cube_impl(dOi_Oi_l1_K1MK0_ed, Di_ub, q_blk_height,
                                                 self.actual_d, precision_type=FP16)

    def compute_dSij(self, Pij_ub, dOi_l1_K1MK0_ed, VjT_K1NK0_ed, Di_ub, kv_blk_height, q_blk_height,
                     dropout_mask_gm_offset):
        """Refer to Algorithm 4 line20 in FlashAttention implement dSij computation"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dPij_ub = self.tik_ops_utils.matmul_compute(dOi_l1_K1MK0_ed, VjT_K1NK0_ed,
                                                        q_blk_height, self.actual_d, kv_blk_height)
            q_blk_height_aligned = self.tik_ops_utils.up_align_to_K0(q_blk_height)
            kv_blk_height_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
            # dropout_mask
            if self.has_drop_mask:
                self.do_dropout_mask(dPij_ub, dropout_mask_gm_offset, kv_blk_height_aligned, kv_blk_height,
                                     q_blk_height_aligned, q_blk_height)
            # dPij - Di
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                broadcast_Di_ub = self.tik_ops_utils.broadcast(Di_ub, (q_blk_height_aligned, kv_blk_height_aligned))
                self.tik_instance.h_sub(dPij_ub, dPij_ub, broadcast_Di_ub)
            self.tik_instance.h_mul(Pij_ub, Pij_ub, dPij_ub)
        return Pij_ub

    def update_dVj(self,
                   PijT_l1_K1MK0_ed,
                   dOi_l1_K1NK0_ed,
                   kv_gm_offset,
                   kv_blk_height,
                   q_blk_height):
        """Refer to Algorithm 4 line16 in FlashAttention implement dVj update"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            PijT_Oi_ub = self.tik_ops_utils.matmul_compute(PijT_l1_K1MK0_ed, dOi_l1_K1NK0_ed,
                                                           kv_blk_height, q_blk_height,
                                                           self.actual_d, precision_type=FP32)
            self.tik_instance.set_atomic_add(1)
            self.cont_data_mv_1_bust(dst=self.dV_gm[kv_gm_offset], src=PijT_Oi_ub,
                                     burst=kv_blk_height * self.d // 8)
            self.tik_instance.set_atomic_add(0)

    def update_dQi(self,
                   dSij_l1_K1MK0_ed,
                   Kj_l1_K1NK0_ed,
                   qo_gm_offset,
                   q_blk_height,
                   kv_blk_height):
        """Refer to Algorithm 4 line21 in FlashAttention implement dQi update"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dSij_Kj_ub = self.tik_ops_utils.matmul_compute(dSij_l1_K1MK0_ed, Kj_l1_K1NK0_ed,
                                                           q_blk_height, kv_blk_height,
                                                           self.actual_d, precision_type=FP32)
            self.tik_instance.set_atomic_add(1)
            self.cont_data_mv_1_bust(dst=self.dQ_gm[qo_gm_offset], src=dSij_Kj_ub,
                                     burst=q_blk_height * self.d // 8)
            self.tik_instance.set_atomic_add(0)

    def update_dKj(self,
                   dSijT_l1_K1MK0_ed,
                   Qi_l1_K1NK0_ed,
                   kv_gm_offset,
                   kv_blk_height,
                   q_blk_height):
        """Refer to Algorithm 4 line22 in FlashAttention implement dKi update"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dSijT_Qi_ub = self.tik_ops_utils.matmul_compute(dSijT_l1_K1MK0_ed, Qi_l1_K1NK0_ed,
                                                            kv_blk_height, q_blk_height,
                                                            self.actual_d, precision_type=FP32)
            self.tik_instance.set_atomic_add(1)
            self.cont_data_mv_1_bust(dst=self.dK_gm[kv_gm_offset], src=dSijT_Qi_ub,
                                     burst=kv_blk_height * self.d // 8)
            self.tik_instance.set_atomic_add(0)

    def compute_in_each_kv_block(self, batch_start, batch_idx, kv_blk_idx, kv_blk_height,
                                 core_idx_to_tr_info, core_idx):
        """The backward computation in each outer loop"""
        kv_blk_height_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
        kv_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.N, self.d,
                                          self.Bc, kv_blk_idx)
        Kj_l1_1 = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Kj_l1_1",
                                           scope=L1)
        Kj_l1_2 = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Kj_l1_2",
                                           scope=L1)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Kj_ub = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Kj_ub", scope=UB)
            self.cont_data_mv_1_bust(dst=Kj_ub, src=self.K_gm[kv_gm_offset],
                                     burst=kv_blk_height * self.d // 16)
            KjT_l1_K1NK0_ed = self.tik_ops_utils.MK_TO_K1MK0(Kj_ub, workspace_tensor=Kj_l1_1)
            Kj_l1_K1NK0_ed = self.tik_ops_utils.KN_TO_K1NK0(Kj_ub, workspace_tensor=Kj_l1_2)

        Vj_l1 = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Vj_l1", scope=L1)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Vj_ub = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Vj_ub", scope=UB)
            self.cont_data_mv_1_bust(dst=Vj_ub, src=self.V_gm[kv_gm_offset],
                                     burst=kv_blk_height * self.d // 16)
            VjT_l1_K1NK0_ed = self.tik_ops_utils.MK_TO_K1MK0(Vj_ub, workspace_tensor=Vj_l1)

        tr_start_s = self.tik_instance.Scalar("int32", name="tr_start_s")
        tr_end_s = self.tik_instance.Scalar("int32", name="tr_end_s")
        tr_start_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 0])
        tr_end_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 1])
        with self.tik_instance.for_range(tr_start_s, tr_end_s, name="q_blk_idx") as q_blk_idx:
            with self.tik_instance.if_scope(tik.all(kv_blk_idx - self.next_block_num <= q_blk_idx,
                                                    q_blk_idx <= kv_blk_idx + self.prev_block_num)):
                with self.tik_instance.if_scope(q_blk_idx != self.Tr - 1):
                    self.compute_in_each_q_block(KjT_l1_K1NK0_ed,
                                                 Kj_l1_K1NK0_ed,
                                                 VjT_l1_K1NK0_ed,
                                                 batch_idx,
                                                 batch_start,
                                                 kv_gm_offset,
                                                 kv_blk_height,
                                                 self.Br,
                                                 kv_blk_idx,
                                                 q_blk_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_q_block(KjT_l1_K1NK0_ed,
                                                 Kj_l1_K1NK0_ed,
                                                 VjT_l1_K1NK0_ed,
                                                 batch_idx,
                                                 batch_start,
                                                 kv_gm_offset,
                                                 kv_blk_height,
                                                 self.last_Br,
                                                 kv_blk_idx,
                                                 q_blk_idx)

    def compute_in_each_q_block(self, KjT_l1_K1NK0_ed, Kj_l1_K1NK0_ed, VjT_l1_K1NK0_ed,
                                batch_idx, batch_start, kv_gm_offset, kv_blk_height,
                                q_blk_height, kv_blk_idx, q_blk_idx):
        """The backward computation in each inner loop"""
        kv_blk_height_alig = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
        q_blk_height_alig = self.tik_ops_utils.up_align_to_K0(q_blk_height)

        Qi_l1_left = self.tik_instance.Tensor(FP16, (q_blk_height_alig, self.d), name="Qi_l1_left",
                                              scope=L1)  # for Qi*KjT (算法11行)
        Qi_l1_right = self.tik_instance.Tensor(FP16, (q_blk_height_alig, self.d), name="Qi_l1_right",
                                               scope=L1)  # for dSijT*Qi (算法22行)
        qo_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Qi_ub = self.tik_instance.Tensor(FP16, (q_blk_height_alig, self.d), name="Qi_ub", scope=UB)
            self.cont_data_mv_1_bust(dst=Qi_ub, src=self.Q_gm[qo_gm_offset],
                                     burst=q_blk_height * self.d // 16)
            Qi_l1_K1MK0_ed = self.tik_ops_utils.MK_TO_K1MK0(Qi_ub, workspace_tensor=Qi_l1_left)
            Qi_l1_K1NK0_ed = self.tik_ops_utils.KN_TO_K1NK0(Qi_ub, workspace_tensor=Qi_l1_right)

        lm_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, 1, self.Br, q_blk_idx)
        attn_mask_gm_offset, dropout_mask_gm_offset, alibi_mask_gm_offset = None, None, None
        if self.has_attn_mask:
            attn_mask_gm_offset = self.get_attn_mask_gm_offset(batch_start, batch_idx, self.Nq, self.N,
                                                               self.Br, q_blk_idx, self.Bc, kv_blk_idx)
        if self.has_drop_mask:
            dropout_mask_gm_offset = self.get_drop_mask_gm_offset(batch_start, batch_idx, self.Nq, self.N,
                                                                  self.Br, q_blk_idx, self.Bc, kv_blk_idx)
        if self.has_alibi_mask:
            alibi_mask_gm_offset = self.get_alibi_gm_offset(batch_start, batch_idx, self.N, self.Bc, kv_blk_idx)
        Pij_ub, Pij_drop_ed_ub = self.compute_Pij(Qi_l1_K1MK0_ed, KjT_l1_K1NK0_ed,
                                                  q_blk_height, self.actual_d, kv_blk_height,
                                                  lm_gm_offset, attn_mask_gm_offset,
                                                  dropout_mask_gm_offset, alibi_mask_gm_offset)

        dOi_l1_right = self.tik_instance.Tensor(FP16, (q_blk_height_alig, self.d), name="dOi_l1_right",
                                                scope=L1)
        dOi_l1_left = self.tik_instance.Tensor(FP16, (q_blk_height_alig, self.d), name="dOi_l1_left",
                                               scope=L1)
        Di_ub = self.tik_instance.Tensor(FP16, (q_blk_height_alig,), name="Di_ub", scope=UB)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dOi_ub = self.tik_instance.Tensor(FP16, (q_blk_height_alig, self.d), name="dOi_ub", scope=UB)
            self.cont_data_mv_1_bust(dst=dOi_ub, src=self.dO_gm[qo_gm_offset],
                                     burst=q_blk_height * self.d // 16)
            self.compute_Di(Di_ub, dOi_ub, qo_gm_offset, q_blk_height)
            dOi_l1_K1NK0_ed = self.tik_ops_utils.KN_TO_K1NK0(dOi_ub, workspace_tensor=dOi_l1_right)
            dOi_l1_K1MK0_ed = self.tik_ops_utils.MK_TO_K1MK0(dOi_ub, workspace_tensor=dOi_l1_left)

        Pij_l1 = self.tik_instance.Tensor(FP16, (q_blk_height_alig, kv_blk_height_alig), name="Pij_l1", scope=L1)
        PijT_l1_K1MK0_ed = self.tik_ops_utils.KN_TO_K1NK0(Pij_drop_ed_ub, workspace_tensor=Pij_l1)
        self.update_dVj(PijT_l1_K1MK0_ed, dOi_l1_K1NK0_ed,
                        kv_gm_offset, kv_blk_height, q_blk_height)
        dSij_l1_1 = self.tik_instance.Tensor(FP16, (q_blk_height_alig, kv_blk_height_alig),
                                             name="dSij_l1_1", scope=L1)
        dSij_l1_2 = self.tik_instance.Tensor(FP16, (q_blk_height_alig, kv_blk_height_alig),
                                             name="dSij_l1_2", scope=L1)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dSij_ub = self.compute_dSij(Pij_ub,
                                        dOi_l1_K1MK0_ed,
                                        VjT_l1_K1NK0_ed,
                                        Di_ub,
                                        kv_blk_height,
                                        q_blk_height,
                                        dropout_mask_gm_offset)
            dSij_l1_K1MK0_ed = self.tik_ops_utils.MK_TO_K1MK0(dSij_ub, workspace_tensor=dSij_l1_1)
            dSijT_l1_K1MK0_ed = self.tik_ops_utils.KN_TO_K1NK0(dSij_ub, workspace_tensor=dSij_l1_2)
        self.update_dQi(dSij_l1_K1MK0_ed, Kj_l1_K1NK0_ed,
                        qo_gm_offset, q_blk_height, kv_blk_height)
        self.update_dKj(dSijT_l1_K1MK0_ed, Qi_l1_K1NK0_ed,
                        kv_gm_offset, kv_blk_height, q_blk_height)

    def compute_one_core(self, batch_start_sc, batch_num_sc, core_idx_to_tr_info, core_idx):
        """The computation of FlashAttention backward on each core"""
        with self.tik_instance.for_range(0, batch_num_sc, name="batch_index") as batch_idx:
            with self.tik_instance.for_range(0, self.Tc, name="kv_blk_idx") as kv_blk_idx:
                with self.tik_instance.if_scope(kv_blk_idx != self.Tc - 1):
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.Bc,
                                                  core_idx_to_tr_info, core_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.last_Bc,
                                                  core_idx_to_tr_info, core_idx)

    def collect_outputs(self):
        """collect all output gm tensors into output_gm_list,
        the output list should keep order with the para order in Primitive and init
        """
        return [self.dQ_gm, self.dK_gm, self.dV_gm]


def flash_attention_grad(Q, K, V, O, dO, l, m, dim_mask, attn_mask, dropout_mask, alibi_mask, dq, dk, dv,
                         prev_block_num=65536,
                         next_block_num=65536,
                         high_precision=False,
                         tiling_stgy_name='xunfei',
                         kernel_name="flash_attention_grad",
                         disable_debug=True):
    """
    algorithm: flash_attention_backward

    Parameters
    ----------
    Q : dict. shape and dtype of input, only support float16
    K : dict. shape and dtype of input, only support float16
    V: dict. shape and dtype of input, only support float16
    O: dict. shape and dtype of input, only support float16
    dO: dict. shape and dtype of input, only support float16
    l: dict. shape and dtype of input, only support float16
    m: dict. shape and dtype of input, only support float16
    dim_mask: dict. shape and dtype of input, only support int8
    dropout_mask: dict. shape and dtype of input, only support float16
    dropout_mask: dict. shape and dtype of input, only support float16
    alibi_mask: dict. shape and dtype of input, only support float16
    dq: dict. shape and dtype of output, only support float16
    dk: dict. shape and dtype of output, only support float16
    dv: dict. shape and dtype of output, only support float16
    prev_block_num: int. an attribute used to define sparse attention
    next_block_num: int. an attribute used to define sparse attention
    tiling_stgy_name: str. an attribute used to choose the tiling strategy
    kernel_name: str. cce kernel name, default value is real_div
    disable_debug: bool. whether disable debug

    Returns
    -------
    tik_instance
    """
    fa_grad = FlashAttentionBwd(Q, K, V, O, dO, l, m, dim_mask, attn_mask, dropout_mask,
                                alibi_mask, prev_block_num=prev_block_num,
                                next_block_num=next_block_num,
                                high_precision=high_precision,
                                kernel_name=kernel_name,
                                tiling_stgy=TilingStrategy.from_strategy_name(tiling_stgy_name),
                                disable_debug=disable_debug)
    fa_grad.compute_process()
    return fa_grad.tik_instance
