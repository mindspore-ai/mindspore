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
"""The forward tik ops of flash attention"""
from collections import defaultdict

from tbe import tik
from tbe.common.platform import set_current_compile_soc_info

from mindspore.ops._op_impl._custom_op.flash_attention.attention import FlashAttention
from mindspore.ops._op_impl._custom_op.flash_attention.constants import FP16
from mindspore.ops._op_impl._custom_op.flash_attention.constants import GM
from mindspore.ops._op_impl._custom_op.flash_attention.constants import L1
from mindspore.ops._op_impl._custom_op.flash_attention.constants import UB
from mindspore.ops._op_impl._custom_op.flash_attention.tiling_strategy.strategy import TilingStrategy

set_current_compile_soc_info("Ascend910")  # 默认310


class FlashAttentionFwd(FlashAttention):
    """The implementation of flash attention forward
    This function contains the flash attention forward implementation used in flash attention (see paper)
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/pdf/2205.14135.pdf>`
    """

    def __init__(self, q, k, v,
                 dim_mask, attn_mask, dropout_mask, alibi_mask,
                 kernel_name,
                 tiling_stgy: TilingStrategy,
                 prev_block_num=65536,
                 next_block_num=65536, disable_debug=True):
        super(FlashAttentionFwd, self).__init__(q, k, v, dim_mask, attn_mask, dropout_mask, alibi_mask, kernel_name,
                                                tiling_stgy, prev_block_num, next_block_num,
                                                disable_debug)
        self.O_gm = None
        self.l_gm = None
        self.m_gm = None

    def define_custom_inputs(self):
        pass

    def define_outputs(self):
        """define output gm tensors"""
        self.O_gm = self.tik_instance.Tensor(FP16, self.O_shape, name="O_gm", scope=GM)
        self.l_gm = self.tik_instance.Tensor(FP16, self.l_shape, name="l_gm", scope=GM)
        self.m_gm = self.tik_instance.Tensor(FP16, self.m_shape, name="m_gm", scope=GM)

    def get_core_bath_info(self):
        """
        Get batch start and batch number of each NPU core.
        :return: Tensor([[core_1_batch_start, core_1_batch_num],...,[core_m_batch_start,
        core_m_batch_num]]), Tensor([[[core_1_batch_1_Tr_start, core_1_batch_1_Tr_end],...[core_1_batch_n_Tr_start,
        core_1_batch_n_Tr_end]],...,[[core_m_batch_1_Tr_start, core_m_batch_1_Tr_end],...[core_m_batch_n_Tr_start,
        core_m_batch_n_Tr_end]]
        """
        if self.core_num > self.B * self.Tr:
            self.core_num = self.B * self.Tr

        task_idx_to_batch_tr_idx = dict()
        for task_idx in range(self.B * self.Tr):
            batch_idx = task_idx // self.Tr
            tr_idx = task_idx % self.Tr
            task_idx_to_batch_tr_idx[task_idx] = [batch_idx, tr_idx]

        core_idx_to_batch_idx = defaultdict(lambda: [100000, -1])
        core_idx_to_tr_idx = defaultdict(lambda: defaultdict(lambda: [100000, -1]))
        task_start = 0
        avg_task_num_per_core, remain_task = divmod(self.B * self.Tr, self.core_num)

        for core_idx in range(self.core_num):
            cur_core_task_num = avg_task_num_per_core
            if core_idx < remain_task:
                cur_core_task_num += 1
            task_end = task_start + cur_core_task_num
            for task_idx in range(task_start, task_end):
                try:
                    batch_idx, tr_idx = task_idx_to_batch_tr_idx[task_idx]
                except KeyError:
                    raise ValueError("The argument 'task_idx' is not valid.")
                # 计算每个核处理的[batch_start, batch_end]
                batch_start_end_pair = core_idx_to_batch_idx[core_idx]
                if batch_idx < batch_start_end_pair[0]:
                    batch_start_end_pair[0] = batch_idx
                if batch_idx > batch_start_end_pair[1]:
                    batch_start_end_pair[1] = batch_idx
                # 计算每个核处理的每个batch的[tr_start, tr_end]
                tr_start_end_pair = core_idx_to_tr_idx[core_idx][batch_idx]
                if tr_idx < tr_start_end_pair[0]:
                    tr_start_end_pair[0] = tr_idx
                if tr_idx > tr_start_end_pair[1]:
                    tr_start_end_pair[1] = tr_idx
            task_start = task_end

        # python dict -> tik tensor
        # [batch_start, batch_idx_end] -> [batch_start, batch_num]
        # [tr_start, tr_idx_end] -> [tr_start, tr_idx_end)
        core_idx_to_batch_info = self.tik_instance.Tensor(
            "int32", (self.core_num, 2), name="core_idx_to_batch_info", scope=UB
        )
        core_idx_to_tr_info = self.tik_instance.Tensor(
            "int32", (self.core_num, self.B, 2), name="core_idx_to_tr_info", scope=UB
        )
        for core_idx in core_idx_to_batch_idx:
            batch_start, batch_end = core_idx_to_batch_idx[core_idx]
            core_idx_to_batch_info[core_idx, 0] = batch_start
            core_idx_to_batch_info[core_idx, 1] = batch_end - batch_start + 1
            for batch_idx in core_idx_to_tr_idx[core_idx]:
                tr_start, tr_end = core_idx_to_tr_idx[core_idx][batch_idx]
                core_idx_to_tr_info[core_idx, batch_idx, 0] = tr_start
                core_idx_to_tr_info[core_idx, batch_idx, 1] = tr_end + 1
        return core_idx_to_batch_info, core_idx_to_tr_info

    def softmax_compute(self, Sij_ub, mij_ub, lij_ub, m, n):
        """Refer to Algorithm 2 line12"""
        # mij = rowmax(Sij) 计算Sij每行的最大值
        self.tik_instance.h_reduce_max(mij_ub, Sij_ub[:, 0:n], 1)
        m_aligned = self.tik_ops_utils.up_align_to_K0(m)
        n_aligned = self.tik_ops_utils.up_align_to_K0(n)
        # Sij - mij
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            broadcast_mij_ub = self.tik_ops_utils.broadcast(mij_ub, (m_aligned, n_aligned))
            self.tik_instance.h_sub(Sij_ub, Sij_ub, broadcast_mij_ub)
        # exp
        self.tik_instance.h_exp(Sij_ub, Sij_ub)

        # cube impl rowsum
        Sij_l1_K1MK0_ws = self.tik_instance.Tensor(FP16, (n_aligned // 16, m_aligned, 16),
                                                   name="Sij_l1_K1MK0_ws", scope=L1)
        Sij_l1_K1MK0_ed = self.tik_ops_utils.MK_TO_K1MK0(Sij_ub, Sij_l1_K1MK0_ws)
        Sij_row_sum_ub = self.tik_ops_utils.row_sum_cube_impl(Sij_l1_K1MK0_ed, lij_ub, m, n)

        return Sij_ub, mij_ub, Sij_row_sum_ub

    def update_m_l(self, mi_old_ub, mij_ub, li_old_ub, lij_ub, vec_len):
        """Refer to Algorithm 2 line13
        mi_new = max(mi, mij), li_new = exp(mi-mi_new)*li + exp(mij - mi_new) * lij
        """
        vec_len_aligned = self.tik_ops_utils.up_align_to_K0(vec_len)
        mi_new_ub = self.tik_instance.Tensor(FP16, (vec_len_aligned,), name="mi_new_ub", scope=UB)
        li_new_ub = self.tik_instance.Tensor(FP16, (vec_len_aligned,), name="li_new_ub", scope=UB)
        # 1 calculate mi_new = max(mi, mij)
        self.tik_instance.h_max(mi_new_ub, mi_old_ub, mij_ub)

        # 2 calculate li_new = exp(mi-mi_new)*li + exp(mij-mi_new)*lij
        # 2.1 相减，求指数
        self.tik_instance.h_sub(mi_old_ub, mi_old_ub, mi_new_ub)  # mi-mi_new
        self.tik_instance.h_exp(mi_old_ub, mi_old_ub)  # exp(mi-mi_new)

        # 2.2 相减，求指数
        self.tik_instance.h_sub(mij_ub, mij_ub, mi_new_ub)  # mij-mi_new
        self.tik_instance.h_exp(mij_ub, mij_ub)  # exp(mij-mi_new)

        # 2.3 相乘，相加 exp(m_ij_ub-mi_new)*li + exp(m_ij_ub-mi_new) * lij
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            mul_li_ub = self.tik_instance.Tensor(FP16, (vec_len_aligned,), scope=UB, name="mul_li_ub")
            mul_lij_ub = self.tik_instance.Tensor(FP16, (vec_len_aligned,), scope=UB, name="mul_lij_ub")
            self.tik_instance.h_mul(mul_li_ub, mi_old_ub, li_old_ub)
            self.tik_instance.h_mul(mul_lij_ub, mij_ub, lij_ub)
            self.tik_instance.h_add(li_new_ub, mul_li_ub, mul_lij_ub)
        return mi_new_ub, li_new_ub

    def update_o_gm(self, block_h, li_new_rec_ub, o_gm_offset, ub_data):
        """Load o from gm and update it, then write it back to gm"""
        half_block_h = block_h // 2
        # double buffer: vec and mte3 parallel
        with self.tik_instance.for_range(0, 2, thread_num=2) as t_idx:
            with self.tik_instance.if_scope(t_idx == 1):
                cur_half_block_h = block_h - half_block_h
            cur_half_block_h_aligned = self.tik_ops_utils.up_align_to_K0(cur_half_block_h)
            row_begin = t_idx * half_block_h
            row_end = row_begin + cur_half_block_h
            broadcast_li_new_rec_ub = self.tik_ops_utils.broadcast(
                li_new_rec_ub[row_begin:row_end], (cur_half_block_h_aligned, self.d)
            )
            self.tik_instance.h_mul(ub_data[row_begin:row_end, :],
                                    ub_data[row_begin:row_end, :],
                                    broadcast_li_new_rec_ub)
            cur_o_gm_offset = o_gm_offset + t_idx * half_block_h * self.d
            self.cont_data_mv_1_bust(dst=self.O_gm[cur_o_gm_offset],
                                     src=ub_data[row_begin:row_end, :],
                                     burst=cur_half_block_h * self.d // 16)

    def update_Oi(
            self,
            Oi_ub,
            exp_mi_sub_mi_new,
            Pij_Vj_ub,
            exp_mij_sub_mi_new,
            li_new_rec_ub,
            li_ub,
            o_gm_offset,
            block_h
    ):
        """Refer to Algorithm 2 line15"""
        block_h_aligned = self.tik_ops_utils.up_align_to_K0(block_h)
        diag_exp_Oi_ub = self.diag_exp_Oi(li_ub, exp_mi_sub_mi_new, Oi_ub, block_h_aligned)
        # exp_mij_sub_mi_new * Pij_Vj_ub
        exp_Pij_Vj_ub = self.exp_Pij_Vj(exp_mij_sub_mi_new, Pij_Vj_ub, block_h_aligned)

        # (diag(li)_exp_Oi + exp_P_V)
        sum_diag_exp_Oi_and_exp_Pij_Vj_ub = diag_exp_Oi_ub
        self.tik_instance.h_add(
            sum_diag_exp_Oi_and_exp_Pij_Vj_ub,
            sum_diag_exp_Oi_and_exp_Pij_Vj_ub,
            exp_Pij_Vj_ub
        )
        self.update_o_gm(block_h, li_new_rec_ub, o_gm_offset, sum_diag_exp_Oi_and_exp_Pij_Vj_ub)

    def diag_exp_Oi(self, li_ub, exp_mi_sub_mi_new, Oi_ub, block_h_aligned):
        """Refer to Algorithm 2 line15
        li * exp(mi - mi_new) * Oi
        """
        self.tik_instance.h_mul(exp_mi_sub_mi_new, exp_mi_sub_mi_new, li_ub)
        diag_exp = exp_mi_sub_mi_new
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            broadcast_diag_exp = self.tik_ops_utils.broadcast(diag_exp, (block_h_aligned, self.d))
            self.tik_instance.h_mul(Oi_ub, Oi_ub, broadcast_diag_exp)
        return Oi_ub

    def exp_Pij_Vj(self, exp_mij_sub_mi_new, Pij_Vj_ub, block_h_aligned):
        """Refer to Algorithm 2 line15
        exp(mij - mi_new) * Pij * Vj
        """
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            broadcast_exp_mij_sub_mi_new = self.tik_ops_utils.broadcast(exp_mij_sub_mi_new, (block_h_aligned, self.d))
            self.tik_instance.h_mul(Pij_Vj_ub, Pij_Vj_ub, broadcast_exp_mij_sub_mi_new)
        return Pij_Vj_ub

    def update_o_m_l(self,
                     Pij_Vj_ub,
                     mij_ub,
                     lij_ub,
                     batch_start,
                     batch_idx,
                     kv_blk_idx,
                     q_blk_idx,
                     block_h):
        """Refer to Algorithm 2 line13 and line15 in FlashAttention"""
        vec_gm_offset = (batch_start + batch_idx) * self.Nq + q_blk_idx * self.Br
        o_gm_offset = self.get_gm_offset(
            batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx
        )
        block_h_aligned = self.tik_ops_utils.up_align_to_K0(block_h)
        # 第一次外循环时，不用走更新逻辑， 减少的无用计算
        # calculate mi_new = mij, li_new=lij. Oi <- Pij*Vj/Li_new
        with self.tik_instance.if_scope(tik.any(kv_blk_idx == 0, kv_blk_idx + self.prev_block_num == q_blk_idx)):
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.l_gm, lij_ub, vec_gm_offset, block_h)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.m_gm, mij_ub, vec_gm_offset, block_h)
            li_new_rec_ub = self.tik_ops_utils.calc_vec_rec(lij_ub, block_h)
            self.update_o_gm(block_h, li_new_rec_ub, o_gm_offset, Pij_Vj_ub)

        with self.tik_instance.else_scope():
            mi_ub = self.tik_instance.Tensor(FP16, (block_h_aligned,), name="mi_old_ub", scope=UB)
            li_ub = self.tik_instance.Tensor(FP16, (block_h_aligned,), name="li_ub", scope=UB)
            self.tik_ops_utils.move_vector_from_gm_to_ub(mi_ub, self.m_gm, vec_gm_offset, block_h)
            self.tik_ops_utils.move_vector_from_gm_to_ub(li_ub, self.l_gm, vec_gm_offset, block_h)

            # 更新 l, m
            mi_new_ub, li_new_ub = self.update_m_l(mi_ub, mij_ub, li_ub, lij_ub, block_h)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.l_gm, li_new_ub, vec_gm_offset, block_h)
            self.tik_ops_utils.move_vector_from_ub_to_gm(self.m_gm, mi_new_ub, vec_gm_offset, block_h)

            exp_mi_sub_mi_new = mi_ub
            exp_mij_sub_mi_new = mij_ub
            # 载入Oi 到 UB
            Oi_ub = self.tik_instance.Tensor(FP16, (block_h_aligned, self.d), scope=UB, name="Oi_ub")
            self.cont_data_mv_1_bust(dst=Oi_ub, src=self.O_gm[o_gm_offset],
                                     burst=block_h * self.d // 16)

            li_new_rec_ub = self.tik_ops_utils.calc_vec_rec(li_new_ub, block_h)
            # 更新O
            self.update_Oi(
                Oi_ub,
                exp_mi_sub_mi_new,
                Pij_Vj_ub,
                exp_mij_sub_mi_new,
                li_new_rec_ub,
                li_ub,
                o_gm_offset,
                block_h
            )

    def compute_in_each_kv_block(self, batch_start, batch_idx, kv_blk_idx, kv_blk_height,
                                 core_idx_to_tr_info, core_idx):
        """The forward computation in each outer loop"""
        kv_blk_height_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
        # load Kj (kv_blk_idx_th block of K_gm), then reorder it for Q*KjT
        Kj_l1 = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Kj_l1", scope=L1)
        kv_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.N, self.d, self.Bc,
                                          kv_blk_idx)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Kj_ub = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Kj_ub", scope=UB)
            self.cont_data_mv_1_bust(dst=Kj_ub, src=self.K_gm[kv_gm_offset],
                                     burst=kv_blk_height * self.d // 16)
            KjT_l1_K1MK0_ed = self.tik_ops_utils.MK_TO_K1MK0(Kj_ub, workspace_tensor=Kj_l1)

        # load Vj (kv_blk_idx_th block of V_gm), then reorder for Pij*Vj
        Vj_l1 = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Vj_l1", scope=L1)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Vj_ub = self.tik_instance.Tensor(FP16, (kv_blk_height_aligned, self.d), name="Vj_ub", scope=UB)
            self.cont_data_mv_1_bust(dst=Vj_ub, src=self.V_gm[kv_gm_offset],
                                     burst=kv_blk_height * self.d // 16)
            Vj_l1_K1NK0_ed = self.tik_ops_utils.KN_TO_K1NK0(Vj_ub, workspace_tensor=Vj_l1)

        tr_start_s = self.tik_instance.Scalar("int32", name="tr_start_s")
        tr_end_s = self.tik_instance.Scalar("int32", name="tr_end_s")
        tr_start_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 0])
        tr_end_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 1])
        with self.tik_instance.for_range(tr_start_s, tr_end_s, name="q_blk_idx") as q_blk_idx:
            # 根据atten_mask倒三角特性，过滤无效计算
            with self.tik_instance.if_scope(tik.all(kv_blk_idx - self.next_block_num <= q_blk_idx,
                                                    q_blk_idx <= kv_blk_idx + self.prev_block_num)):
                with self.tik_instance.if_scope(q_blk_idx != self.Tr - 1):
                    self.compute_in_each_q_block(KjT_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx,
                                                 batch_start,
                                                 kv_blk_height, self.Br, q_blk_idx, kv_blk_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_q_block(KjT_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx,
                                                 batch_start,
                                                 kv_blk_height, self.last_Br, q_blk_idx, kv_blk_idx)

    def compute_in_each_q_block(self, KjT_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx, batch_start,
                                kv_blk_height, q_blk_height, q_blk_idx, kv_blk_idx):
        """The forward computation in each inner loop"""
        kv_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_height)
        q_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(q_blk_height)
        # load Qi (q_blk_idx_th block of Q_gm), then reorder it fo Qi*KjT
        Qi_l1 = self.tik_instance.Tensor(FP16, (q_blk_h_aligned, self.d), scope=L1, name="Qi_l1")
        q_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Qi_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned, self.d), scope=UB, name="Qi_ub")
            self.cont_data_mv_1_bust(dst=Qi_ub, src=self.Q_gm[q_gm_offset],
                                     burst=q_blk_height * self.d // 16)
            Qi_l1_K1MK0_ed = self.tik_ops_utils.MK_TO_K1MK0(Qi_ub, workspace_tensor=Qi_l1)

        lij_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned,), scope=UB, name="lij_ub")
        mij_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned,), scope=UB, name="mij_ub")
        Pij_l1_K1MK0_ed = self.tik_instance.Tensor(
            FP16, (kv_blk_h_aligned // 16, q_blk_h_aligned, 16), name="Pij_l1", scope=L1
        )
        # QK^T Q shape: (q_blk_h_aligned, self.d), K^T shape: (self.d, kv_blk_h_aligned)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Sij_ub_MN_ed = self.tik_ops_utils.matmul_compute(Qi_l1_K1MK0_ed, KjT_l1_K1MK0_ed, m=q_blk_height,
                                                             k=self.actual_d, n=kv_blk_height,
                                                             N1MN0_to_MN=True)  # Qi*KjT
            if self.has_alibi_mask:
                alibi_mask_gm_offset = self.get_alibi_gm_offset(batch_start, batch_idx, self.N, self.Bc, kv_blk_idx)
                self.do_alibi_mask(Sij_ub_MN_ed, alibi_mask_gm_offset, q_blk_h_aligned, kv_blk_h_aligned)

            # att_mask
            if self.has_attn_mask:
                attn_mask_gm_offset = self.get_attn_mask_gm_offset(batch_start, batch_idx, self.Nq, self.N,
                                                                   self.Br, q_blk_idx, self.Bc, kv_blk_idx)
                self.do_att_mask(Sij_ub_MN_ed, attn_mask_gm_offset, q_blk_height, kv_blk_height,
                                 q_blk_h_aligned, kv_blk_h_aligned)

            Pij_ub, mij_ub, lij_ub = self.softmax_compute(
                Sij_ub_MN_ed, mij_ub, lij_ub, q_blk_height, kv_blk_height
            )
            # dropout_mask
            if self.has_drop_mask:
                dropout_mask_gm_offset = self.get_drop_mask_gm_offset(batch_start, batch_idx, self.Nq,
                                                                      self.N, self.Br, q_blk_idx, self.Bc, kv_blk_idx)
                self.do_dropout_mask(Pij_ub, dropout_mask_gm_offset, kv_blk_h_aligned, kv_blk_height,
                                     q_blk_h_aligned, q_blk_height)

            # ub -> l1
            Pij_l1_K1MK0_ed = self.tik_ops_utils.MK_TO_K1MK0(Pij_ub, workspace_tensor=Pij_l1_K1MK0_ed)
        # Pij_Vj_ub shape:             (M, K) (q_blk_h_aligned, kv_blk_h_aligned)
        # Vj_l1 shape:                 (K,N)  (kv_blk_h_aligned, self.d)
        # Pij_Vj_matmul_res_ub shape:  (M, N) (q_blk_h_aligned, self.d)
        Pij_Vj_matmul_res_ub = self.tik_ops_utils.matmul_compute(Pij_l1_K1MK0_ed, Vj_l1_K1NK0_ed, q_blk_height,
                                                                 kv_blk_height, self.actual_d,
                                                                 N1MN0_to_MN=True)  # Pij*Vj
        self.update_o_m_l(
            Pij_Vj_matmul_res_ub,
            mij_ub,
            lij_ub,
            batch_start,
            batch_idx,
            kv_blk_idx,
            q_blk_idx,
            q_blk_height
        )

    def compute_one_core(self, batch_start_sc, batch_num_sc, core_idx_to_tr_info, core_idx):
        """The computation of FlashAttention forward on each core"""
        with self.tik_instance.for_range(0, batch_num_sc, name="batch_index") as batch_idx:
            with self.tik_instance.for_range(0, self.Tc, name="kv_blk_idx") as kv_blk_idx:
                with self.tik_instance.if_scope(kv_blk_idx != self.Tc - 1):
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.Bc,
                                                  core_idx_to_tr_info, core_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.last_Bc,
                                                  core_idx_to_tr_info, core_idx)

    def collect_inputs(self):
        """collect all input gm tensors into input_gm_list,
        the input list should keep order with the para order in Primitive and init
        """
        input_gm_list = [self.Q_gm, self.K_gm, self.V_gm, self.dim_mask_gm]
        if self.has_attn_mask:
            input_gm_list.append(self.att_mask_gm)
        if self.has_drop_mask:
            input_gm_list.append(self.drop_mask_gm)
        if self.has_alibi_mask:
            input_gm_list.append(self.alibi_mask_gm)

        return input_gm_list

    def collect_outputs(self):
        """collect all output gm tensors into output_gm_list,
        the output list should keep order with the para order in Primitive and init
        """
        return [self.O_gm, self.l_gm, self.m_gm]

    def compute_process(self):
        """The compute process of FlashAttention forward"""
        self.init()

        core_idx_to_batch_info, core_idx_to_tr_info = self.get_core_bath_info()
        with self.tik_instance.for_range(begint=0, endt=self.core_num, name="core_index",
                                         block_num=self.core_num) as core_idx:
            batch_start_s = self.tik_instance.Scalar("int32", name="batch_start_s")
            batch_num_s = self.tik_instance.Scalar("int32", name="batch_num_s")

            batch_start_s.set_as(core_idx_to_batch_info[core_idx, 0])
            batch_num_s.set_as(core_idx_to_batch_info[core_idx, 1])

            self.compute_one_core(batch_start_s, batch_num_s, core_idx_to_tr_info, core_idx)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=self.collect_inputs(),
            outputs=self.collect_outputs(),
            config={"dump_cce_code": False, "save_temp_cce_file": True, "enable_const_fold": True},
            enable_l2=True
        )


def flash_attention(q, k, v, dim_mask, attn_mask, dropout_mask, alibi_mask, y, l, m,
                    prev_block_num=65536, next_block_num=65536, tiling_stgy_name='xunfei',
                    kernel_name="flash_attention", disable_debug=True):
    """
    algorithm: flash_attention_backward

    Parameters
    ----------
    q : dict. shape and dtype of input, only support float16
    k : dict. shape and dtype of input, only support float16
    v: dict. shape and dtype of input, only support float16
    dim_mask: dict. shape and dtype of input, only support int8
    attn_mask: dict. shape and dtype of input, only support float16
    dropout_mask: dict. shape and dtype of input, only support float16
    dropout_mask: dict. shape and dtype of input, only support float16
    alibi_mask: dict. shape and dtype of input, only support float16
    y: dict. shape and dtype of output, only support float16
    l: dict. shape and dtype of output, only support float16
    m: dict. shape and dtype of output, only support float16
    prev_block_num: int. an attribute used to define sparse attention
    next_block_num: int. an attribute used to define sparse attention
    tiling_stgy_name: str. an attribute used to choose the tiling strategy
    kernel_name: str. cce kernel name, default value is real_div
    disable_debug: bool. whether disable debug

    Returns
    -------
    tik_instance
    """
    fa = FlashAttentionFwd(q=q, k=k, v=v, dim_mask=dim_mask, attn_mask=attn_mask,
                           dropout_mask=dropout_mask, alibi_mask=alibi_mask, kernel_name=kernel_name,
                           tiling_stgy=TilingStrategy.from_strategy_name(tiling_stgy_name),
                           prev_block_num=prev_block_num, next_block_num=next_block_num,
                           disable_debug=disable_debug)
    fa.compute_process()
    return fa.tik_instance
