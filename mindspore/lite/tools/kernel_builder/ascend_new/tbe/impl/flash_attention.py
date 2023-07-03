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
"""ascend custom op: flash attention by tik"""
import math
from collections import defaultdict
import te.platform as tbe_platform
from tbe import tik
from tbe.common.platform import get_soc_spec
from tbe.common.platform import set_current_compile_soc_info

set_current_compile_soc_info("Ascend910")
BLOCK_NUM = 16
FP16 = "float16"
INT8 = "int8"
INT32 = "int32"
FP32 = "float32"
REPEAT_SZ = 128
BLK_STRIDE = 1
REPEAT_STRIDE = 8
TRANS_CUBE_TGT = 8
FP16_MIN_VAL = -65504.0

GM = tik.scope_gm
L1 = tik.scope_cbuf
L1OUT = tik.scope_cbuf_out
UB = tik.scope_ubuf
L0A = tik.scope_ca
L0B = tik.scope_cb
L0C = tik.scope_cc

DTYPE_SIZE = {
    "int8": 1,
    "float16": 2,
    "int16": 2,
    "float32": 4,
}

SOFTMAX_WITH_ROWMAX = True
UPDATE_SUB_CORE_STRATEGY = True


class FlashAttention:
    """
    custom op: flash attention
    """

    def __init__(self, q, k, v, mask, disable_debug=True):
        """
        :param q: q tensor
        :param k: k tensor
        :param v: v tensor
        :param mask: mask for tensor
        :param disable_debug: for debug
        """
        self.tik_instance = tik.Tik(disable_debug=disable_debug)
        self.core_num = get_soc_spec(tbe_platform.CORE_NUM)
        self.l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
        self.update_sub_core_strategy = UPDATE_SUB_CORE_STRATEGY

        if isinstance(q, dict):
            self.q_shape = q["shape"]  # [B, Nq, d], B = batch_size * head_num
            self.k_shape = k["shape"]  # [B, N, d]
            self.v_shape = v["shape"]  # [B, N, dv]
            self.mask_shape = mask["shape"]
        else:
            self.q_shape = q.shape  # [B, Nq, d], B = batch_size * head_num
            self.k_shape = k.shape  # [B, N, d]
            self.v_shape = v.shape  # [B, N, dv]
            self.mask_shape = mask.shape

        self.b_dim, self.nq_dim, self.d = self.q_shape
        self.n_dim = self.k_shape[1]

        self.l_shape = [self.b_dim, self.nq_dim]
        self.m_shape = [self.b_dim, self.nq_dim]
        self.o_shape = [self.b_dim, self.nq_dim, self.d]
        self.actual_d = self.q_shape[-1] if self.mask_shape[0] == 1 else self.mask_shape[0]

        self.dtype = "float16"
        self.k0 = 16

    def tiling_for_fa(self):
        """
        tiling for flash attention fusion
        """
        if self.d == 512:
            self.bc = 128
            self.br = 32
        elif self.n_dim == 14144 or self.n_dim == 10560:
            self.bc = 256
            self.br = 64
        elif self.n_dim < 512:
            self.bc = 128
            self.br = 128
        else:
            self.bc = 512
            self.br = 32
        self.tc = math.ceil(self.n_dim / self.bc)
        self.tr = math.ceil(self.nq_dim / self.br)
        self.last_bc = self.bc if self.n_dim % self.bc == 0 else self.n_dim % self.bc
        self.last_br = self.br if self.nq_dim % self.br == 0 else self.nq_dim % self.br

        self.tik_instance.tikdb.debug_print('"self.bc:", self.bc')
        self.tik_instance.tikdb.debug_print('"self.br:", self.br')
        self.tik_instance.tikdb.debug_print('"self.tr:", self.tr')
        self.tik_instance.tikdb.debug_print('"self.tc:", self.tc')
        self.tik_instance.tikdb.debug_print('"self.last_br:", self.last_br')
        self.tik_instance.tikdb.debug_print('"self.last_bc:", self.last_bc')

    def init(self):
        """
        Init for flash attention op.
        """
        self.tiling_for_fa()
        self.init_gm()

    def init_gm(self):
        """
        Init for tensor
        """
        # define inputs
        self.q_gm = self.tik_instance.Tensor(FP16, self.q_shape, name="q_gm", scope=GM)
        self.k_gm = self.tik_instance.Tensor(FP16, self.k_shape, name="k_gm", scope=GM)
        self.v_gm = self.tik_instance.Tensor(FP16, self.v_shape, name="v_gm", scope=GM)
        self.mask_gm = self.tik_instance.Tensor(INT8, self.mask_shape, name="mask_gm", scope=GM)

        # define output and intermediate res
        self.o_gm = self.tik_instance.Tensor(FP16, self.o_shape, name="o_gm", scope=GM)
        self.l_gm = self.tik_instance.Tensor(
            FP16, self.l_shape, name="l_gm", scope=GM, is_workspace=True
        )
        self.m_gm = self.tik_instance.Tensor(
            FP16, self.m_shape, name="m_gm", scope=GM, is_workspace=True
        )

    def get_core_bath_info(self):
        """Get batch start and batch number of each NPU core.

        :return: Tensor([core_1_batch_start, core_1_batch_num,.....core_n_batch_start,
        core_n_batch_num])
        """
        core_batch_info = self.tik_instance.Tensor(
            INT32, (self.core_num * 2,), name="batch_info", scope=UB
        )
        core_batch_start = 0
        avg_batch_num_per_core, remain_batch = divmod(self.b_dim, self.core_num)
        for core_idx in range(self.core_num):
            cur_core_batch_num = avg_batch_num_per_core
            if core_idx < remain_batch:
                cur_core_batch_num += 1

            core_batch_info[2 * core_idx] = core_batch_start
            core_batch_info[2 * core_idx + 1] = cur_core_batch_num
            core_batch_start += cur_core_batch_num

        return core_batch_info

    def get_each_core_task_info(self):
        """Get task info for each core.
        """
        task_idx_to_batch_tr_idx = dict()
        for task_idx in range(self.b_dim * self.tr):
            batch_idx = task_idx // self.tr
            tr_idx = task_idx % self.tr
            task_idx_to_batch_tr_idx[task_idx] = [batch_idx, tr_idx]

        core_idx_to_batch_idx = defaultdict(lambda: [100000, -1])
        core_idx_to_tr_idx = defaultdict(lambda: defaultdict(lambda: [100000, -1]))
        task_start = 0
        avg_task_num_per_core, remain_task = divmod(self.b_dim * self.tr, self.core_num)

        for core_idx in range(self.core_num):
            cur_core_task_num = avg_task_num_per_core
            if core_idx < remain_task:
                cur_core_task_num += 1
            task_end = task_start + cur_core_task_num
            for task_idx in range(task_start, task_end):
                batch_idx, tr_idx = task_idx_to_batch_tr_idx[task_idx]  # batch_idx: 0~16, tr_idx: 0-128
                batch_start_end_pair = core_idx_to_batch_idx[core_idx]  # [10000, -1]
                if batch_idx < batch_start_end_pair[0]:
                    batch_start_end_pair[0] = batch_idx
                if batch_idx > batch_start_end_pair[1]:
                    batch_start_end_pair[1] = batch_idx
                tr_start_end_pair = core_idx_to_tr_idx[core_idx][batch_idx]  # [10000, -1]
                if tr_idx < tr_start_end_pair[0]:
                    tr_start_end_pair[0] = tr_idx
                if tr_idx > tr_start_end_pair[1]:
                    tr_start_end_pair[1] = tr_idx
            task_start = task_end
        core_idx_to_batch_info = self.tik_instance.Tensor(
            "int32", (self.core_num, 2), name="core_idx_to_batch_info", scope=UB
        )
        core_idx_to_tr_info = self.tik_instance.Tensor(
            "int32", (self.core_num, self.b_dim, 2), name="core_idx_to_tr_info", scope=UB
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

    def get_gm_offset(self, batch_start, batch_idx, h, w, block_h, block_idx):
        """
        Get tensor offset
        """
        gm_offset = (batch_start + batch_idx) * h * w + block_idx * block_h * w
        return gm_offset

    def move_vector_from_gm_to_ub(self, dst_tensor, src_tensor, gm_offset):
        """load the vector from gm to ub
        :param dst_tensor:
        :param src_tensor:
        :param gm_offset:
        :return:
        """
        vec_len = dst_tensor.shape[0]
        full_tik_blk_num, tail_num = divmod(vec_len, 16)
        with self.tik_instance.if_scope(full_tik_blk_num > 0):
            self.tik_instance.data_move(
                dst_tensor, src_tensor[gm_offset], 0, 1, full_tik_blk_num, 0, 0
            )
        with self.tik_instance.if_scope(tail_num > 0):
            offset = vec_len - 16
            last_blk_ub = self.tik_instance.Tensor(FP16, (16,), name="last_blk_ub", scope=UB)
            self.tik_instance.data_move(last_blk_ub, src_tensor[gm_offset + offset], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, 16) as idx:
                dst_tensor[offset + idx].set_as(last_blk_ub[idx])

    def move_vector_from_ub_to_gm(self, dst_tensor, src_tensor, gm_offset, block_h):
        """write the vector back to gm
        :param dst_tensor:
        :param src_tensor:
        :param gm_offset:
        :param block_h:
        :return:
        """
        full_tik_blk_num = block_h // 16
        with self.tik_instance.if_scope(full_tik_blk_num > 0):
            self.tik_instance.data_move(
                dst_tensor[gm_offset], src_tensor, 0, 1, full_tik_blk_num, 0, 0
            )
        tail_num = block_h % 16
        with self.tik_instance.if_scope(tail_num > 0):
            offset = block_h - 16
            tmp_ub = self.tik_instance.Tensor(FP16, (16,), name="tmp_ub", scope=UB)
            with self.tik_instance.for_range(0, 16) as idx:
                tmp_ub[idx].set_as(src_tensor[offset + idx])
            self.tik_instance.data_move(dst_tensor[gm_offset + offset], tmp_ub, 0, 1, 1, 0, 0)

    def mk_to_k1mk0(self, mk_input_tensor, workspace_tensor=None):
        """change data shape from (up_align_m, K) to (k1, up_align_m, k0), k1 = K // k0, the effect is equant to:
        new_tensor =  np.stack(np.hsplit(mk_input_tensor, k1), axis=0)

        :param mk_input_tensor: input tensor in GM with shape: (up_align_m, K)
        :param workspace_tensor: workspace tensor with shape: (k1, up_align_m, k0)
        tensor will be changed, otherwise the new data will be copied to the workspace tensor,
        and input tensor will stay unchanged.
        :return:
        """
        dtype = mk_input_tensor.dtype
        m, k = mk_input_tensor.shape
        k0 = 16
        k1 = k // k0
        up_align_m = self.up_align_to_k0(m)
        if workspace_tensor is not None:
            with self.tik_instance.for_range(0, k1) as i:
                self.tik_instance.data_move(
                    workspace_tensor[i * up_align_m * k0:],
                    mk_input_tensor[i * k0:],
                    0,
                    up_align_m,
                    k0 * DTYPE_SIZE[dtype] // 32,
                    (k1 - 1) * k0 * DTYPE_SIZE[dtype] // 32,
                    0,
                )
            return workspace_tensor.reshape((k1, up_align_m, k0))
        else:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                tmp_ub = self.tik_instance.Tensor(dtype, (k1, up_align_m, k0), name="tmp_ub", scope=UB)
                # data_move(m,k) --> (k1,m,k0)
                with self.tik_instance.for_range(0, k1) as i:
                    self.tik_instance.data_move(
                        tmp_ub[i * up_align_m * k0:],
                        mk_input_tensor[i * k0:],
                        0,
                        up_align_m,
                        k0 * DTYPE_SIZE[dtype] // 32,
                        (k1 - 1) * k0 * DTYPE_SIZE[dtype] // 32,
                        0,
                    )
                self.tik_instance.data_move(
                    mk_input_tensor, tmp_ub, 0, 1, k1 * up_align_m * k0 * DTYPE_SIZE[dtype] // 32, 0, 0
                )
                return mk_input_tensor.reshape((k1, up_align_m, k0))

    def transpose_matrix(self, src_ub, dst_ub, n, nk0=False):
        """ transpose matrix, default support shape: (16, n) -> (n, 16)
        if nk0 is true, support shape: (n, 16) -> (16, n)
        """
        k0 = 16
        rep_times = n // k0
        if nk0:
            src_list = [src_ub[16 * i] for i in range(16)]
            dst_list = [dst_ub[n * i] for i in range(16)]
        else:
            src_list = [src_ub[n * i] for i in range(16)]
            dst_list = [dst_ub[16 * i] for i in range(16)]

        dst_rep_stride = k0
        src_rep_stride = 1
        if rep_times == 1:
            dst_rep_stride = 0
            src_rep_stride = 0

        if nk0:
            src_rep_stride, dst_rep_stride = dst_rep_stride, src_rep_stride

        self.tik_instance.vec_trans_scatter(
            False, False, dst_list, src_list, rep_times, dst_rep_stride, src_rep_stride
        )
        return dst_ub

    def kn_to_k1nk0(self, kn_input_tensor, workspace_tensor=None):
        """change data shape from (K,N) to (k1, N, k0), k1 = K // k0, the effect is equant to:
        new_tensor =  np.reshape(kn_input_tensor, newshape=(k1, k0, N)).swapaxes(1, 2)

        :param kn_input_tensor: input tensor with shape: (K, N)
        :param workspace_tensor: workspace tensor with shape: (k1, N, k0)
        tensor will be changed, otherwise the new data will be copied to the workspace tensor,
        and input tensor will stay unchanged.
        :return:
        """
        dtype = kn_input_tensor.dtype
        k, n = kn_input_tensor.shape
        k0 = 16
        k1 = k // k0
        up_n = n
        with self.tik_instance.for_range(0, k1) as index:
            k1nk0_ub = self.tik_instance.Tensor(dtype, (up_n, k0), UB, "k1nk0_ub")
            src_ub = self.tik_instance.Tensor(dtype, (k0, up_n), UB, "src_ub")
            burst_len = k0 * up_n * DTYPE_SIZE[dtype] // 32
            self.tik_instance.data_move(
                src_ub, kn_input_tensor[index * k0 * up_n], 0, 1, burst_len, 0, 0
            )
            k1nk0_ub = self.transpose_matrix(src_ub, k1nk0_ub, up_n)
            if workspace_tensor is None:
                self.tik_instance.data_move(
                    kn_input_tensor[index * k0 * up_n], k1nk0_ub, 0, 1, burst_len, 0, 0
                )
            else:
                self.tik_instance.data_move(
                    workspace_tensor[index * k0 * up_n], k1nk0_ub, 0, 1, burst_len, 0, 0
                )
        if workspace_tensor is None:
            return kn_input_tensor.reshape((k1, up_n, k0))
        else:
            return workspace_tensor.reshape((k1, up_n, k0))

    def kn_to_k1nk0_v2(self, kn_input_tensor, workspace_tensor):
        """change data shape from (K,N) to (k1, N, k0), k1 = K // k0, the effect is equant to:
        new_tensor =  np.reshape(kn_input_tensor, newshape=(k1, k0, N)).swapaxes(1, 2)

        :param kn_input_tensor: input tensor with shape: (K, N)
        :param workspace_tensor: workspace tensor with shape: (k1, N, k0)
        tensor will be changed, otherwise the new data will be copied to the workspace tensor,
        and input tensor will stay unchanged.
        :return:
        """
        dtype = kn_input_tensor.dtype
        k, n = kn_input_tensor.shape
        k0 = 16
        k1 = k // k0
        up_n = n
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            workspace_tensor_tmp = self.tik_instance.Tensor(dtype, (k, n), UB, "workspace_tensor_tmp")
            # k1k0n -> k0k1n
            with self.tik_instance.for_range(0, k0) as index:
                self.tik_instance.data_move(workspace_tensor_tmp[index * k1 * up_n], kn_input_tensor[index * up_n],
                                            0, k1, up_n // 16, (k0 - 1) * up_n // 16, 0)
            # k0(k1n) -> k1nk0
            kn_input_tensor = self.transpose_matrix(workspace_tensor_tmp, kn_input_tensor, k1 * up_n)
        # ub -> l1
        self.tik_instance.data_move(workspace_tensor, kn_input_tensor, 0, 1, k * n // 16, 0, 0)
        return workspace_tensor.reshape((k1, up_n, k0))

    def n1mn0_to_mn(self, n1mn0_input):
        """change data shape from (N1, up_align_m, N0) to (up_align_m, N), N0=16, N = N1 * k0, the effect is equant to:
        n1mn0_input = np.concatenate(list(map(np.squeeze, np.split(n1mn0_input, N1))), axis=1)

        :param n1mn0_input: input tensor with shape (N, up_align_m, N0) in GM or L1.
        :return:
        """
        dtype = n1mn0_input.dtype
        n1, up_align_m, n0 = n1mn0_input.shape

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            tmp_ub = self.tik_instance.Tensor(dtype, (up_align_m, n1 * n0), name="tmp_ub", scope=UB)
            # data_move (n1,m,n0) --> (m,n)
            with self.tik_instance.for_range(0, n1) as i:
                self.tik_instance.data_move(
                    tmp_ub[i * n0:],
                    n1mn0_input[i * up_align_m * n0:],
                    0,
                    up_align_m,
                    n0 * DTYPE_SIZE[dtype] // 32,
                    0,
                    (n1 - 1) * n0 * DTYPE_SIZE[dtype] // 32,
                )
            # data_move out
            self.tik_instance.data_move(
                n1mn0_input, tmp_ub, 0, 1, up_align_m * n1 * n0 * DTYPE_SIZE[dtype] // 32, 0, 0
            )
        return n1mn0_input.reshape((up_align_m, n1 * n0))

    def matmul_compute(self, a_l1, b_l1, c_ub, m, k, n, k0=16, reorder_res=True):
        """calculate matrix multiplication a_l1 * b_l1, and move the result to c_ub,
        then rearrange c_ub
        :param a_l1: input tensor in L1 with shape of (k_alig // 16, m_alig, 16)
        :param b_l1: input tensor in L1 with shape of (k_alig // 16, n_alig, 16)
        :param c_ub: workspace tensor in UB with shape of (n_alig // 16, m_alig, 16)
        :param m: the actual number of rows of a_l1
        :param k: the actual number of cols of a_l1
        :param n: the actual number of cols of b_l1
        :param k0: matrix fractal param
        :return: c_ub with tensor with shape of (m_alig, n_alig)
        """
        m_alig = self.up_align_to_k0(m)
        n_alig = self.up_align_to_k0(n)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # matmul
            c_l0c = self.tik_instance.Tensor(
                FP32, (n_alig // k0, m_alig, k0), scope=L0C, name="c_l0c"
            )  # n1mn0 (n0=16)
            self.tik_instance.matmul(c_l0c, a_l1, b_l1, m, k, n)
            t_num = 2
            if m_alig * n_alig == 256:
                t_num = 1
            with self.tik_instance.for_range(0, t_num, thread_num=t_num) as t_id:
                tmp_c_ub_fp32 = self.tik_instance.Tensor(FP32, (m_alig * n_alig // t_num,), name="tmp_c_ub_fp32",
                                                         scope=UB)
                tmp_c_ub_fp16 = self.tik_instance.Tensor(FP16, (m_alig * n_alig // t_num,), name="tmp_c_ub_fp16",
                                                         scope=UB)
                offset = t_id * (m_alig * n_alig // t_num)
                self.tik_instance.data_move(tmp_c_ub_fp32,
                                            c_l0c[offset],
                                            0,
                                            1,
                                            m_alig * n_alig // t_num * DTYPE_SIZE[FP32] // 1024,
                                            0,
                                            0)
                self.tik_instance.h_cast(tmp_c_ub_fp16, tmp_c_ub_fp32, "none")
                self.tik_instance.data_move(c_ub[offset],
                                            tmp_c_ub_fp16,
                                            0,
                                            1,
                                            m_alig * n_alig // t_num * DTYPE_SIZE[FP16] // 32,
                                            0,
                                            0)
        if reorder_res:
            return self.n1mn0_to_mn(c_ub)
        else:
            return c_ub

    def scale_compute_vector(self, sij_ub, dim):
        """
        compute scale vector tensor
        """
        scale_value = dim ** -0.5
        scale = self.tik_instance.Scalar(dtype=FP16)
        scale.set_as(scale_value)
        self.tik_instance.h_mul(sij_ub, sij_ub, scale)
        return sij_ub

    def softmax_compute(self, sij_ub, mij_ub, lij_ub, m, n):
        """
        sij_ub shape up_align_m x N
        """
        self.tik_instance.h_reduce_max(mij_ub, sij_ub[0:m, 0:n], 1)
        # Sij - mij
        with self.tik_instance.for_range(0, m) as i:
            src_scalar = self.tik_instance.Scalar(init_value=mij_ub[i], dtype=FP16)
            self.tik_instance.h_sub(sij_ub[i, :], sij_ub[i, :], src_scalar)
        # exp
        self.tik_instance.h_exp(sij_ub, sij_ub)

        # cube impl rowsum
        m_alig = self.up_align_to_k0(m)
        n_alig = self.up_align_to_k0(n)
        sij_l1_k1mk0_ed = self.tik_instance.Tensor(FP16, (n_alig // 16, m_alig, 16), name="sij_l1_k1mk0_ed", scope=L1)
        sij_l1_k1mk0_ed = self.mk_to_k1mk0(sij_ub, sij_l1_k1mk0_ed)
        lij_ub = self.row_sum_cube_impl(sij_l1_k1mk0_ed, lij_ub, m, n)

        return sij_ub, mij_ub, lij_ub

    def update_m_l(self, mi_old_ub, mij_ub, li_old_ub, lij_ub, vec_len):
        """
        update mi and li tensor
        """
        mi_new_ub = self.tik_instance.Tensor(FP16, (vec_len,), name="mi_new_ub", scope=UB)
        li_new_ub = self.tik_instance.Tensor(FP16, (vec_len,), name="li_new_ub", scope=UB)
        self.tik_instance.h_max(mi_new_ub, mi_old_ub, mij_ub)

        self.tik_instance.h_sub(mi_old_ub, mi_old_ub, mi_new_ub)  # mi-mi_new
        self.tik_instance.h_exp(mi_old_ub, mi_old_ub)  # exp(mi-mi_new)

        self.tik_instance.h_sub(mij_ub, mij_ub, mi_new_ub)  # mij-mi_new
        self.tik_instance.h_exp(mij_ub, mij_ub)  # exp(mij-mi_new)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            mul_li_ub = self.tik_instance.Tensor(FP16, (vec_len,), scope=UB, name="mul_li_ub")
            mul_lij_ub = self.tik_instance.Tensor(FP16, (vec_len,), scope=UB, name="mul_lij_ub")
            self.tik_instance.h_mul(mul_li_ub, mi_old_ub, li_old_ub)
            self.tik_instance.h_mul(mul_lij_ub, mij_ub, lij_ub)
            self.tik_instance.h_add(li_new_ub, mul_li_ub, mul_lij_ub)
        return mi_new_ub, li_new_ub

    def clac_new_oi(
            self,
            oi_ub,
            exp_mi_sub_mi_new,
            pij_vj_ub,
            exp_mij_sub_mi_new,
            li_new_rec_ub,
            li_ub,
            block_h,
            block_w,
    ):
        """Oi_new = (l_i_old * exp(m_i_old - m_i_new) * Oi_old + exp(m_ij - m_i_new) * Pij*Vj) / l_i_new
        :param oi_ub:
        :param exp_mi_sub_mi_new:
        :param pij_vj_ub: (q_blk_h_aligned, self.d)
        :param exp_mij_sub_mi_new:
        :param li_new_rec_ub:
        :param li_ub:
        :param block_h:
        :param block_w:
        :return:
        """
        diag_exp_oi_ub = self.diag_exp_oi(li_ub, exp_mi_sub_mi_new, oi_ub, block_h, block_w)

        exp_pij_vj_ub = self.exp_pij_vj(exp_mij_sub_mi_new, pij_vj_ub, block_h, block_w)

        sum_diag_exp_oi_and_exp_pij_vj_ub = diag_exp_oi_ub
        self.tik_instance.h_add(
            sum_diag_exp_oi_and_exp_pij_vj_ub,
            sum_diag_exp_oi_and_exp_pij_vj_ub,
            exp_pij_vj_ub
        )
        with self.tik_instance.for_range(begint=0, endt=block_h) as i:
            src_scalar = self.tik_instance.Scalar(init_value=li_new_rec_ub[i], dtype=FP16)
            self.tik_instance.h_mul(
                sum_diag_exp_oi_and_exp_pij_vj_ub[i, :],
                src_scalar,
                sum_diag_exp_oi_and_exp_pij_vj_ub[i, :],
            )

        return sum_diag_exp_oi_and_exp_pij_vj_ub

    def diag_exp_oi(self, li_ub, exp_mi_sub_mi_new, oi_ub, block_h, block_w):
        """
        compute oi tensor
        """
        self.tik_instance.h_mul(exp_mi_sub_mi_new, exp_mi_sub_mi_new, li_ub)
        diag_exp = exp_mi_sub_mi_new
        with self.tik_instance.for_range(begint=0, endt=block_h) as i:
            src_scalar = self.tik_instance.Scalar(init_value=diag_exp[i], dtype=FP16)
            self.tik_instance.h_mul(oi_ub[i, :], oi_ub[i, :], src_scalar)
        return oi_ub

    def exp_pij_vj(self, exp_mij_sub_mi_new, pij_vj_ub, block_h, block_w):
        """
        compute pij_vj_ub tensor
        """
        with self.tik_instance.for_range(begint=0, endt=block_h) as i:
            src_scalar = self.tik_instance.Scalar(init_value=exp_mij_sub_mi_new[i], dtype=FP16)
            self.tik_instance.h_mul(pij_vj_ub[i, :], src_scalar, pij_vj_ub[i, :])
        return pij_vj_ub

    def calc_vec_rec(self, li_new_ub, block_h):
        """
        compute rec tensor
        """
        li_new_rec_ub = self.tik_instance.Tensor(FP16, (block_h,), scope=UB, name="li_new_rec_ub")
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            repeat_times = block_h // 128
            if repeat_times > 0:
                mask_len = 128
                dst_rep_stride = 8
                src_rep_stride = 8
                block_len = 16  # src dtype is float16
                src_extent_size = (repeat_times - 1) * src_rep_stride * block_len + mask_len
                wk_size_unit = ((src_extent_size + block_len - 1) // block_len) * block_len
                wk_size = 4 * wk_size_unit
                work_tensor_ub = self.tik_instance.Tensor(
                    "float32", (wk_size,), name="work_tensor_ub", scope=UB
                )
                self.tik_instance.vec_rec_high_preci(
                    mask_len,
                    li_new_rec_ub[0:],
                    li_new_ub[0:],
                    work_tensor_ub[0:],
                    repeat_times,
                    dst_rep_stride,
                    src_rep_stride,
                )

            mask_len = block_h - repeat_times * 128
            if mask_len > 0:
                wk_size = 4 * ((mask_len + 16 - 1) // 16) * 16
                work_tensor_ub2 = self.tik_instance.Tensor(
                    "float32", (wk_size,), name="work_tensor_ub2", scope=UB
                )
                self.tik_instance.vec_rec_high_preci(
                    mask_len,
                    li_new_rec_ub[repeat_times * 128:],
                    li_new_ub[repeat_times * 128:],
                    work_tensor_ub2[0:],
                    1,
                    0,
                    0,
                )
        return li_new_rec_ub

    def update_o_m_l(self,
                     pij_vj_ub,
                     mij_ub,
                     lij_ub,
                     batch_start,
                     batch_idx,
                     kv_blk_idx,
                     q_blk_idx,
                     block_h):
        """ load o m l from gm and update them in ub, then write them back to gm
        :param pij_vj_ub: input tensor with shape of (q_blk_h_aligned, self.d)
        :param mij_ub: input tensor with shape of (Br)
        :param lij_ub: input tensor with shape of (Br)
        :param batch_start:
        :param batch_idx:
        :param kv_blk_idx:
        :param q_blk_idx:
        :param block_h:
        :return: None
        """
        vec_gm_offset = (batch_start + batch_idx) * self.nq_dim + q_blk_idx * self.br
        o_gm_offset = self.get_gm_offset(
            batch_start, batch_idx, self.nq_dim, self.d, self.br, q_blk_idx
        )
        h_alig = self.up_align_to_k0(block_h)
        with self.tik_instance.if_scope(kv_blk_idx == 0):
            self.move_vector_from_ub_to_gm(self.l_gm, lij_ub, vec_gm_offset, block_h)
            self.move_vector_from_ub_to_gm(self.m_gm, mij_ub, vec_gm_offset, block_h)
            li_new_rec_ub = self.calc_vec_rec(lij_ub, block_h)
            with self.tik_instance.for_range(begint=0, endt=block_h) as i:
                src_scalar = self.tik_instance.Scalar(init_value=li_new_rec_ub[i], dtype=FP16)
                self.tik_instance.h_mul(
                    pij_vj_ub[i, :],
                    src_scalar,
                    pij_vj_ub[i, :],
                )
            self.tik_instance.data_move(self.o_gm[o_gm_offset], pij_vj_ub, 0, 1, block_h * self.d // 16, 0, 0)
        with self.tik_instance.else_scope():
            mi_ub = self.tik_instance.Tensor(FP16, (block_h,), name="mi_old_ub", scope=UB)
            li_ub = self.tik_instance.Tensor(FP16, (block_h,), name="li_ub", scope=UB)
            self.move_vector_from_gm_to_ub(mi_ub, self.m_gm, vec_gm_offset)
            self.move_vector_from_gm_to_ub(li_ub, self.l_gm, vec_gm_offset)

            mi_new_ub, li_new_ub = self.update_m_l(mi_ub, mij_ub, li_ub, lij_ub, block_h)
            exp_mi_sub_mi_new = mi_ub
            exp_mij_sub_mi_new = mij_ub
            oi_ub = self.tik_instance.Tensor(FP16, (h_alig, self.d), scope=UB, name="oi_ub")
            self.tik_instance.data_move(
                oi_ub, self.o_gm[o_gm_offset], 0, 1, block_h * self.d // 16, 0, 0
            )

            li_new_rec_ub = self.calc_vec_rec(li_new_ub, block_h)
            oi_new_ub = self.clac_new_oi(
                oi_ub,
                exp_mi_sub_mi_new,
                pij_vj_ub,
                exp_mij_sub_mi_new,
                li_new_rec_ub,
                li_ub,
                block_h,
                self.d,
            )

            self.tik_instance.data_move(self.o_gm[o_gm_offset], oi_new_ub, 0, 1, block_h * self.d // 16, 0, 0)
            self.move_vector_from_ub_to_gm(self.l_gm, li_new_ub, vec_gm_offset, block_h)
            self.move_vector_from_ub_to_gm(self.m_gm, mi_new_ub, vec_gm_offset, block_h)

    def row_sum_cube_impl(self, sij_l1_k1mk0_ed, lij_ub, m, k):
        """
        :param sij_ub: the tensor with shape of (k1, up_align_m, k0) in ub
        :return:
        """
        k1, up_align_m, k0 = sij_l1_k1mk0_ed.shape
        up_k = k1 * k0

        right_all_one_matrix_ub = self.tik_instance.Tensor(
            FP16, (up_k, 16), name="right_all_one_matrix_ub", scope=UB
        )
        self.tik_instance.h_duplicate(right_all_one_matrix_ub, 1.0)
        right_all_one_matrix_l1 = self.tik_instance.Tensor(
            FP16, (k1 * k0, 16), name="right_all_one_matrix_l1", scope=L1
        )
        self.tik_instance.data_move(right_all_one_matrix_l1,
                                    right_all_one_matrix_ub,
                                    0, 1, up_k, 0, 0)

        row_sum_ub = self.tik_instance.Tensor(FP16, (1, up_align_m, 16), name="row_sum_ub", scope=UB)
        row_sum_ub_n1mn0 = self.matmul_compute(sij_l1_k1mk0_ed,
                                               right_all_one_matrix_l1,
                                               row_sum_ub,
                                               m, k, 16, reorder_res=False)
        row_sum_ub_mn_ed = row_sum_ub_n1mn0.reshape((up_align_m, 16))
        row_sum_ub_trans = self.tik_instance.Tensor(FP16, (16, up_align_m), name="row_sum_ub_trans", scope=UB)
        row_sum_ub_trans = self.transpose_matrix(row_sum_ub_mn_ed, row_sum_ub_trans, up_align_m, True)
        self.tik_instance.data_move(lij_ub, row_sum_ub_trans[0, 0:m], 0, 1, m // 16, 0, 0)

        return lij_ub

    def softmax_without_row_max(self, sij_ub, lij_ub, m, n):
        """
        :param sij_ub:
        :param lij_ub:
        :return:
        """
        k1, up_align_m, k0 = sij_ub.shape
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self.tik_instance.h_exp(sij_ub, sij_ub)

            sij_l1_k1mk0_ed = self.tik_instance.Tensor(FP16, (k1, up_align_m, k0), name="sij_l1_k1mk0_ed",
                                                       scope=L1)
            self.tik_instance.data_move(sij_l1_k1mk0_ed, sij_ub, 0, 1, k1 * up_align_m * k0 // 16, 0, 0)
            lij_ub = self.row_sum_cube_impl(sij_l1_k1mk0_ed, lij_ub, m, n)

        return sij_ub, lij_ub

    def update_l_o_without_rowmax(self,
                                  pij_vj_ub,
                                  lij_ub,
                                  batch_start,
                                  batch_idx,
                                  kv_blk_idx,
                                  q_blk_idx,
                                  block_h
                                  ):
        """
        update l_o tensor
        """
        vec_gm_offset = (batch_start + batch_idx) * self.nq_dim + q_blk_idx * self.br
        o_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.nq_dim, self.d, self.br, q_blk_idx)
        h_alig = self.up_align_to_k0(block_h)
        with self.tik_instance.if_scope(kv_blk_idx == 0):
            self.move_vector_from_ub_to_gm(self.l_gm, lij_ub, vec_gm_offset, block_h)
            li_new_rec_ub = self.calc_vec_rec(lij_ub, block_h)
            with self.tik_instance.for_range(begint=0, endt=2, thread_num=2) as t_id:
                half_block_h = self.tik_instance.Scalar("int32", init_value=block_h // 2)
                with self.tik_instance.if_scope(t_id == 1):
                    half_block_h.set_as(block_h - half_block_h)
                half_block_offset = t_id * (block_h // 2)
                with self.tik_instance.for_range(begint=0, endt=half_block_h) as i:
                    src_scalar = self.tik_instance.Scalar(init_value=li_new_rec_ub[half_block_offset + i], dtype=FP16)
                    self.tik_instance.h_mul(
                        pij_vj_ub[half_block_offset + i, :],
                        src_scalar,
                        pij_vj_ub[half_block_offset + i, :],
                    )
                o_half_block_offset = o_gm_offset + half_block_offset * self.d
                tensor = pij_vj_ub[half_block_offset, :]
                self.tik_instance.data_move(self.o_gm[o_half_block_offset], tensor, 0, 1,
                                            half_block_h * self.d // 16, 0, 0)

        with self.tik_instance.else_scope():
            li_ub = self.tik_instance.Tensor(FP16, (block_h,), name="li_ub", scope=UB)
            self.move_vector_from_gm_to_ub(li_ub, self.l_gm, vec_gm_offset)
            self.tik_instance.h_add(lij_ub, li_ub, lij_ub)
            self.move_vector_from_ub_to_gm(self.l_gm, lij_ub, vec_gm_offset, block_h)

            with self.tik_instance.new_stmt_scope(disable_sync=False):
                oi_ub = self.tik_instance.Tensor(FP16, (h_alig, self.d), scope=UB, name="oi_ub")
                self.tik_instance.data_move(
                    oi_ub, self.o_gm[o_gm_offset], 0, 1, block_h * self.d // 16, 0, 0
                )
                with self.tik_instance.for_range(0, block_h) as i:
                    src_scalar = self.tik_instance.Scalar(init_value=li_ub[i], dtype=FP16)
                    self.tik_instance.h_mul(oi_ub[i, :], oi_ub[i, :], src_scalar)

                self.tik_instance.h_add(pij_vj_ub, oi_ub, pij_vj_ub)

            li_new_rec_ub = self.calc_vec_rec(lij_ub, block_h)
            with self.tik_instance.for_range(begint=0, endt=2, thread_num=2) as t_id:
                half_block_h = self.tik_instance.Scalar("int32", init_value=block_h // 2)
                with self.tik_instance.if_scope(t_id == 1):
                    half_block_h.set_as(block_h - half_block_h)
                half_block_offset = t_id * (block_h // 2)
                with self.tik_instance.for_range(begint=0, endt=half_block_h) as i:
                    src_scalar = self.tik_instance.Scalar(
                        init_value=li_new_rec_ub[half_block_offset + i], dtype=FP16
                    )
                    self.tik_instance.h_mul(
                        pij_vj_ub[half_block_offset + i, :],
                        src_scalar,
                        pij_vj_ub[half_block_offset + i, :],
                    )
                o_half_block_offset = o_gm_offset + half_block_offset * self.d
                src_tensor = pij_vj_ub[half_block_offset, :]
                self.tik_instance.data_move(self.o_gm[o_half_block_offset], src_tensor, 0, 1,
                                            half_block_h * self.d // 16, 0, 0)

    def up_align_to_k0(self, n, dtype=None):
        """
        up align for k0 tensor.
        """
        if dtype is None:
            dtype = self.dtype

        k0 = 32 // DTYPE_SIZE[dtype]
        return (n + k0 - 1) // k0 * k0

    def compute_in_each_kv_block(self, batch_start, batch_idx, kv_blk_idx, kv_blk_height,
                                 core_idx_to_tr_info=None, core_idx=None):
        """
        compute kv_block tensor.
        """
        bc_alig = self.up_align_to_k0(kv_blk_height)
        # gm -> ub ->l1
        kj_l1_k1mk0_ed = self.tik_instance.Tensor(FP16, (bc_alig, self.d), name="Kj_l1", scope=L1)
        k_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.n_dim, self.d,
                                         self.bc, kv_blk_idx)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            kj_ub = self.tik_instance.Tensor(FP16, (bc_alig, self.d), name="kj_ub", scope=UB)
            self.tik_instance.data_move(
                kj_ub, self.k_gm[k_gm_offset], 0, 1, kv_blk_height * self.d // 16, 0, 0
            )
            kj_l1_k1mk0_ed = self.mk_to_k1mk0(kj_ub, workspace_tensor=kj_l1_k1mk0_ed)

        # kn_to_k1nk0: gm -> ub -> for (src_ub -> k1nk0_ub -> l1)
        # kn_to_k1nk0_v2: gm -> ub -> ub -> scatter_ub -> l1
        vj_l1_k1nk0_ed = self.tik_instance.Tensor(FP16, (bc_alig, self.d), name="Vj_l1", scope=L1)
        v_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.n_dim, self.d,
                                         self.bc, kv_blk_idx)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            vj_ub = self.tik_instance.Tensor(FP16, (bc_alig, self.d), name="vj_ub", scope=UB)
            self.tik_instance.data_move(
                vj_ub, self.v_gm[v_gm_offset], 0, 1, kv_blk_height * self.d // 16, 0, 0
            )
            vj_l1_k1nk0_ed = self.kn_to_k1nk0(vj_ub, workspace_tensor=vj_l1_k1nk0_ed)

        if self.update_sub_core_strategy:
            tr_start_s = self.tik_instance.Scalar("int32", name="tr_start_s")
            tr_end_s = self.tik_instance.Scalar("int32", name="tr_end_s")
            tr_start_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 0])
            tr_end_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 1])
            with self.tik_instance.for_range(tr_start_s, tr_end_s, name="q_blk_idx") as q_blk_idx:
                with self.tik_instance.if_scope(q_blk_idx != self.tr - 1):
                    self.compute_in_each_q_block(kj_l1_k1mk0_ed, vj_l1_k1nk0_ed, batch_idx,
                                                 batch_start,
                                                 kv_blk_height, self.br, q_blk_idx, kv_blk_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_q_block(kj_l1_k1mk0_ed, vj_l1_k1nk0_ed, batch_idx,
                                                 batch_start,
                                                 kv_blk_height, self.last_br, q_blk_idx, kv_blk_idx)
        else:
            # for double buffer.
            with self.tik_instance.for_range(0, self.tr // 2, name="q_blk_idx") as q_loop_idx:
                with self.tik_instance.for_range(begint=0, endt=2, thread_num=2) as t_id:
                    q_blk_idx = 2 * q_loop_idx + t_id
                    with self.tik_instance.if_scope(q_blk_idx != self.tr - 1):
                        self.compute_in_each_q_block(kj_l1_k1mk0_ed, vj_l1_k1nk0_ed, batch_idx,
                                                     batch_start,
                                                     kv_blk_height, self.br, q_blk_idx, kv_blk_idx)
                    with self.tik_instance.else_scope():
                        self.compute_in_each_q_block(kj_l1_k1mk0_ed, vj_l1_k1nk0_ed, batch_idx,
                                                     batch_start,
                                                     kv_blk_height, self.last_br, q_blk_idx, kv_blk_idx)

    def compute_in_each_q_block(self, kj_l1_k1mk0_ed, vj_l1_k1nk0_ed, batch_idx, batch_start,
                                kv_blk_height, q_blk_height, q_blk_idx, kv_blk_idx):
        """
        compute kv_block tensor.
        """
        kv_blk_h_aligned = self.up_align_to_k0(kv_blk_height)
        q_blk_h_aligned = self.up_align_to_k0(q_blk_height)
        # up_align_m,K
        # gm -> ub -> l1
        qi_l1_k1mk0_ed = self.tik_instance.Tensor(FP16, (q_blk_h_aligned, self.d), scope=L1, name="Qi_l1")
        q_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.nq_dim, self.d, self.br, q_blk_idx)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            qi_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned, self.d), scope=UB, name="qi_ub")
            self.tik_instance.data_move(
                qi_ub, self.q_gm[q_gm_offset], 0, 1, q_blk_height * self.d // 16, 0, 0
            )
            qi_l1_k1mk0_ed = self.mk_to_k1mk0(qi_ub, workspace_tensor=qi_l1_k1mk0_ed)

        if SOFTMAX_WITH_ROWMAX:
            mij_ub = self.tik_instance.Tensor(FP16, (q_blk_height,), scope=UB, name="mij_ub")
        lij_ub = self.tik_instance.Tensor(FP16, (q_blk_height,), scope=UB, name="lij_ub")
        pij_l1_k1mk0_ed = self.tik_instance.Tensor(
            FP16, (kv_blk_h_aligned // 16, q_blk_h_aligned, 16), name="Pij_l1", scope=L1
        )
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # N1MN0
            sij_ub = self.tik_instance.Tensor(FP16, (kv_blk_h_aligned // 16, q_blk_h_aligned, 16), name="sij_ub",
                                              scope=UB)
            # QK^T Q shape: (q_blk_h_aligned, self.d), K^T shape: (self.d, kv_blk_h_aligned)

            reorder_res = True
            if not SOFTMAX_WITH_ROWMAX:
                reorder_res = False
            sij_ub_mn_ed = self.matmul_compute(
                qi_l1_k1mk0_ed,
                kj_l1_k1mk0_ed,
                sij_ub,
                m=q_blk_height,
                k=self.actual_d,
                n=kv_blk_height,
                reorder_res=reorder_res
            )  # q*kT

            sij_ub_mn_ed_scaled = self.scale_compute_vector(sij_ub_mn_ed, self.actual_d)

            if SOFTMAX_WITH_ROWMAX:
                pij_ub, mij_ub, lij_ub = self.softmax_compute(sij_ub_mn_ed_scaled, mij_ub, lij_ub, q_blk_height,
                                                              kv_blk_height)
            else:
                pij_ub, lij_ub = self.softmax_without_row_max(sij_ub_mn_ed_scaled, lij_ub, q_blk_height, kv_blk_height)
            # ub -> l1
            if reorder_res:
                pij_l1_k1mk0_ed = self.mk_to_k1mk0(pij_ub, workspace_tensor=pij_l1_k1mk0_ed)
            else:
                self.tik_instance.data_move(pij_l1_k1mk0_ed,
                                            pij_ub,
                                            0,
                                            1,
                                            q_blk_h_aligned * kv_blk_h_aligned // 16,
                                            0,
                                            0)

        pij_vj_ub = self.tik_instance.Tensor(FP16, (self.d // 16, q_blk_h_aligned, 16), name="pij_vj_ub", scope=UB)
        # pij_vj_ub shape:             (up_align_m, K) (q_blk_h_aligned, kv_blk_h_aligned)
        # Vj_l1 shape:                 (K,N)  (kv_blk_h_aligned, self.d)
        # pij_vj_matmul_res_ub shape:  (up_align_m, N) (q_blk_h_aligned, self.d)
        pij_vj_matmul_res_ub = self.matmul_compute(
            pij_l1_k1mk0_ed, vj_l1_k1nk0_ed, pij_vj_ub, q_blk_height, kv_blk_height, self.actual_d,
            reorder_res=True
        )
        if SOFTMAX_WITH_ROWMAX:
            self.update_o_m_l(
                pij_vj_matmul_res_ub,
                mij_ub,
                lij_ub,
                batch_start,
                batch_idx,
                kv_blk_idx,
                q_blk_idx,
                q_blk_height
            )
        else:
            self.update_l_o_without_rowmax(
                pij_vj_matmul_res_ub,
                lij_ub,
                batch_start,
                batch_idx,
                kv_blk_idx,
                q_blk_idx,
                q_blk_height
            )

    def compute_one_core(self, batch_start_sc, batch_num_sc, core_idx_to_tr_info=None, core_idx=None):
        """
        compute tensor for one core.
        """
        with self.tik_instance.for_range(0, batch_num_sc, name="batch_index") as batch_idx:
            with self.tik_instance.for_range(0, self.tc, name="kv_blk_idx") as kv_blk_idx:
                with self.tik_instance.if_scope(kv_blk_idx != self.tc - 1):
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.bc,
                                                  core_idx_to_tr_info, core_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.last_bc,
                                                  core_idx_to_tr_info, core_idx)

    def compute_process(self):
        """
        compute process.
        """
        self.init()
        if self.update_sub_core_strategy:
            if self.core_num > self.b_dim * self.tr:
                self.core_num = self.b_dim * self.tr
            core_idx_to_batch_info, core_idx_to_tr_info = self.get_each_core_task_info()
            with self.tik_instance.for_range(begint=0, endt=self.core_num, name="core_index",
                                             block_num=self.core_num) as core_idx:
                batch_start_s = self.tik_instance.Scalar("int32", name="batch_start_s")
                batch_num_s = self.tik_instance.Scalar("int32", name="batch_num_s")
                batch_start_s.set_as(core_idx_to_batch_info[core_idx, 0])
                batch_num_s.set_as(core_idx_to_batch_info[core_idx, 1])
                self.compute_one_core(batch_start_s, batch_num_s, core_idx_to_tr_info, core_idx)
        else:
            if self.core_num > self.b_dim:
                self.core_num = self.b_dim
            core_batch_info = self.get_core_bath_info()
            with self.tik_instance.for_range(begint=0, endt=self.core_num, name="core_index",
                                             block_num=self.core_num) as core_idx:
                batch_start, batch_num = (
                    core_batch_info[2 * core_idx],
                    core_batch_info[2 * core_idx + 1],
                )
                batch_start_s = self.tik_instance.Scalar(dtype=INT32, name='batch_start_s')
                batch_num_s = self.tik_instance.Scalar(dtype=INT32, name='batch_num_s')

                batch_start_s.set_as(batch_start[0])
                batch_num_s.set_as(batch_num[0])

                self.compute_one_core(batch_start_s, batch_num_s)


def flash_attention(q, k, v, mask, y, kernel_name="flash_attention", disable_debug=True):
    """
    define flash attention custom op.
    """
    fa = FlashAttention(q=q, k=k, v=v, mask=mask, disable_debug=disable_debug)
    fa.compute_process()
    fa.tik_instance.BuildCCE(
        kernel_name=kernel_name,
        inputs=[fa.q_gm, fa.k_gm, fa.v_gm, fa.mask_gm],
        outputs=[fa.o_gm],
        config={"dump_cce_code": False, "save_temp_cce_file": True, "enable_const_fold": True},
        enable_l2=True
    )

    return fa.tik_instance
