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
"""the common about tik ops"""
from functools import partial

from mindspore.ops._op_impl._custom_op.flash_attention.constants import DTYPE_SIZE
from mindspore.ops._op_impl._custom_op.flash_attention.constants import FP16
from mindspore.ops._op_impl._custom_op.flash_attention.constants import FP32
from mindspore.ops._op_impl._custom_op.flash_attention.constants import L0C
from mindspore.ops._op_impl._custom_op.flash_attention.constants import UB


class TikOpsUtils:
    """Utils function class about tik ops"""

    def __init__(self, tik_instance):
        self.tik_instance = tik_instance
        self.dtype = "float16"
        self.cont_data_mv_1_bust = partial(self.tik_instance.data_move, sid=0, nburst=1,
                                           src_stride=0,
                                           dst_stride=0)

    def MK_TO_K1MK0(self, mk_input_tensor, workspace_tensor=None):
        """change data shape from (M, K) to (K1, M, K0), K1 = K // K0, the effect is equant to:
        new_tensor =  np.stack(np.hsplit(mk_input_tensor, K1), axis=0)

        :param mk_input_tensor: input tensor in GM with shape: (M, K)
        :param workspace_tensor: workspace tensor with shape: (K1, M, K0)
        tensor will be changed, otherwise the new data will be copied to the workspace tensor,
        and input tensor will stay unchanged.
        :return: Tensor with shape (K1,M, K0)
        """
        dtype = mk_input_tensor.dtype
        m, k = mk_input_tensor.shape
        K0 = 16
        K1 = k // K0
        M = self.up_align_to_K0(m)
        try:
            dtype_size = DTYPE_SIZE[dtype]
        except KeyError:
            raise ValueError("The argument 'dtype' is not valid.")
        if workspace_tensor is not None:
            with self.tik_instance.for_range(0, K1) as i:
                self.tik_instance.data_move(
                    workspace_tensor[i * M * K0:],
                    mk_input_tensor[i * K0:],
                    0,
                    M,
                    K0 * dtype_size // 32,
                    (K1 - 1) * K0 * dtype_size // 32,
                    0,
                )
            return workspace_tensor.reshape((K1, M, K0))

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            tmp_ub = self.tik_instance.Tensor(dtype, (K1, M, K0), name="tmp_ub", scope=UB)
            # data_move(m,k) --> (k1,m,K0)
            with self.tik_instance.for_range(0, K1) as i:
                self.tik_instance.data_move(
                    tmp_ub[i * M * K0:],
                    mk_input_tensor[i * K0:],
                    0,
                    M,
                    K0 * dtype_size // 32,
                    (K1 - 1) * K0 * dtype_size // 32,
                    0,
                )
            self.cont_data_mv_1_bust(
                dst=mk_input_tensor, src=tmp_ub, burst=K1 * M * K0 * dtype_size // 32)
            return mk_input_tensor.reshape((K1, M, K0))

    def transpose_matrix(self, src_ub, dst_ub, N, nk0=False):
        """ transpose matrix, default support shape: (16, n) -> (n, 16)
        if nk0 is true, support shape: (n, 16) -> (16, n)
        """
        K0 = 16
        rep_times = N // K0
        if nk0:
            src_list = [src_ub[16 * i] for i in range(16)]
            dst_list = [dst_ub[N * i] for i in range(16)]
        else:
            src_list = [src_ub[N * i] for i in range(16)]
            dst_list = [dst_ub[16 * i] for i in range(16)]

        dst_rep_stride = K0
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

    def KN_TO_K1NK0(self, kn_input_tensor, workspace_tensor=None):
        """change data shape from (K,N) to (K1, N, K0), K1 = K // K0, the effect is equvilent to:
        new_tensor =  np.reshape(kn_input_tensor, newshape=(K1, K0, N)).swapaxes(1, 2)

        :param kn_input_tensor: input tensor with shape: (K, N)
        :param workspace_tensor: workspace tensor with shape: (K1, N, K0)
        tensor will be changed, otherwise the new data will be copied to the workspace tensor,
        and input tensor will stay unchanged.
        :return: Tensor with shape: (K1, N, K0)
        """
        dtype = kn_input_tensor.dtype
        k, n = kn_input_tensor.shape
        K0 = 16
        K1 = k // K0
        N = n
        try:
            dtype_size = DTYPE_SIZE[dtype]
        except KeyError:
            raise ValueError("The argument 'dtype' is not valid.")
        with self.tik_instance.for_range(0, K1) as index:
            k1nk0_ub = self.tik_instance.Tensor(dtype, (N, K0), UB, "k1nk0_ub")
            src_ub = self.tik_instance.Tensor(dtype, (K0, N), UB, "src_ub")
            burst_len = K0 * N * dtype_size // 32
            self.cont_data_mv_1_bust(dst=src_ub, src=kn_input_tensor[index * K0 * N],
                                     burst=burst_len)
            k1nk0_ub = self.transpose_matrix(src_ub, k1nk0_ub, N)
            if workspace_tensor is None:
                self.cont_data_mv_1_bust(dst=kn_input_tensor[index * K0 * N], src=k1nk0_ub,
                                         burst=burst_len)
            else:
                self.cont_data_mv_1_bust(dst=workspace_tensor[index * K0 * N], src=k1nk0_ub,
                                         burst=burst_len)
        if workspace_tensor is None:
            return kn_input_tensor.reshape((K1, N, K0))

        return workspace_tensor.reshape((K1, N, K0))

    def N1MN0_TO_MN(self, N1MN0_input):
        """change data shape from (N1, M, N0) to (M, N), N0=16, N = N1 * K0, the effect is equant to:
        N1MN0_input = np.concatenate(list(map(np.squeeze, np.split(N1MN0_input, N1))), axis=1)

        :param N1MN0_input: input tensor with shape (N, M, N0) in GM or L1.
        :return:
        """
        dtype = N1MN0_input.dtype
        N1, M, N0 = N1MN0_input.shape
        try:
            dtype_size = DTYPE_SIZE[dtype]
        except KeyError:
            raise ValueError("The argument 'dtype' is not valid.")
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            tmp_ub = self.tik_instance.Tensor(dtype, (M, N1 * N0), name="tmp_ub", scope=UB)
            # data_move (n1,m,n0) --> (m,n)
            with self.tik_instance.for_range(0, N1) as i:
                self.tik_instance.data_move(
                    tmp_ub[i * N0:],
                    N1MN0_input[i * M * N0:],
                    0,
                    M,
                    N0 * dtype_size // 32,
                    0,
                    (N1 - 1) * N0 * dtype_size // 32,
                )
            # data_move out
            self.cont_data_mv_1_bust(dst=N1MN0_input, src=tmp_ub, burst=M * N1 * N0 * dtype_size // 32)
        return N1MN0_input.reshape((M, N1 * N0))

    def broadcast(self, vec_ub, shape):
        """ broadcast a vector to a matrix
        :param vec_ub: a tensor in UB with shape of (M,), and dtype is float16
        :param shape: the target shape, a tuple with value (M, N), M and N are integer multiples of 16
        :return: a tensor in UB with shape of (M, N)
        """
        M, N = shape
        dst_ub = self.tik_instance.Tensor(FP16, shape, name="dst_ub", scope=UB)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # (M,) -> (2, M) -> (4, M) -> (8, M) -> (16, M)
            tmp_ub1 = self.tik_instance.Tensor(FP16, (16, M), name="tmp_ub1", scope=UB)
            self.tik_instance.data_move(tmp_ub1, vec_ub, 0, 1, M // 16, 0, 0)
            times = self.tik_instance.Scalar("int32", name="times", init_value=1)
            with self.tik_instance.for_range(begint=0, endt=16):
                with self.tik_instance.if_scope(times <= 8):
                    offset = times * M
                    burst = times * M // 16
                    self.cont_data_mv_1_bust(dst=tmp_ub1[offset], src=tmp_ub1, burst=burst)
                with self.tik_instance.else_scope():
                    self.tik_instance.tik_break()
                times.set_as(times * 2)

            # (16, M) -> (M, 16)
            tmp_ub2 = self.tik_instance.Tensor(FP16, (M, 16), name="tmp_ub2", scope=UB)
            tmp_ub2_transposed = self.transpose_matrix(tmp_ub1, tmp_ub2, M)

            # (M, 16) -> (M, 32) -> (M, 64) -> ... -> (M, N)
            self.tik_instance.data_move(dst_ub, tmp_ub2_transposed, 0, M, 1, 0, N // 16 - 1)
            times.set_as(1)
            with self.tik_instance.for_range(begint=0, endt=N):
                offset = times * 16
                with self.tik_instance.if_scope(offset * 2 <= N):
                    burst = offset // 16
                    src_stride = N // 16 - burst
                    dst_stride = N // 16 - burst
                    self.tik_instance.data_move(dst_ub[offset], dst_ub, 0, M, burst, src_stride,
                                                dst_stride)
                with self.tik_instance.else_scope():
                    burst = (N - offset) // 16
                    src_stride = N // 16 - burst
                    dst_stride = N // 16 - burst
                    with self.tik_instance.if_scope(burst > 0):
                        self.tik_instance.data_move(dst_ub[offset], dst_ub, 0, M, burst, src_stride,
                                                    dst_stride)
                    self.tik_instance.tik_break()
                times.set_as(times * 2)
        return dst_ub

    def broadcast_row(self, vec_ub, shape):
        """broadcast row"""
        M, N = shape
        dst_ub = self.tik_instance.Tensor(FP16, shape, name="dst_ub", scope=UB)
        self.tik_instance.data_move(dst_ub, vec_ub, 0, 1, N // 16, 0, 0)
        times = self.tik_instance.Scalar("int32", name="times", init_value=1)
        # (1, N) -> (2, M) -> (4, N) -> ... -> (M, N)
        with self.tik_instance.for_range(begint=0, endt=M):
            with self.tik_instance.if_scope(times * 2 <= M):
                burst = times * N // 16
                offset = times * N
                self.tik_instance.data_move(dst_ub[offset], dst_ub, 0, 1, burst, 0, 0)
            with self.tik_instance.else_scope():
                burst = (M - times) * N // 16
                offset = times * N
                with self.tik_instance.if_scope(burst > 0):
                    self.tik_instance.data_move(dst_ub[offset], dst_ub, 0, 1, burst, 0, 0)
                self.tik_instance.tik_break()
            times.set_as(times * 2)
        return dst_ub

    def get_K0(self, dtype=None):
        """get K0"""
        if dtype is None:
            dtype = self.dtype
        try:
            dtype_size = DTYPE_SIZE[dtype]
        except KeyError:
            raise ValueError("The argument 'dtype' is not valid.")
        return 32 // dtype_size

    def up_align_to_K0(self, n, dtype=None):
        """byte alignment by dtype"""
        if dtype is None:
            dtype = self.dtype
        try:
            dtype_size = DTYPE_SIZE[dtype]
        except KeyError:
            raise ValueError("The argument 'dtype' is not valid.")
        K0 = 32 // dtype_size
        return (n + K0 - 1) // K0 * K0

    def calc_vec_rec(self, vec_ub, vec_len):
        """cal the reciprocal of a vector"""
        dtype = vec_ub.dtype
        vec_len_aligned = self.up_align_to_K0(vec_len)
        vec_rec_ub = self.tik_instance.Tensor(dtype, (vec_len_aligned,), scope=UB, name="li_new_rec_ub")
        try:
            dtype_size = DTYPE_SIZE[dtype]
        except KeyError:
            raise ValueError("The argument 'dtype' is not valid.")
        mask_len = 256 // dtype_size
        block_len = 32 // dtype_size
        work_size = 8 // dtype_size

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            repeat_times = vec_len // mask_len
            if repeat_times > 0:
                dst_rep_stride = 8
                src_rep_stride = 8

                src_extent_size = (repeat_times - 1) * src_rep_stride * block_len + mask_len
                wk_size_unit = ((src_extent_size + block_len - 1) // block_len) * block_len
                wk_size = work_size * wk_size_unit
                # 定义work_tensor
                work_tensor_ub = self.tik_instance.Tensor(
                    "float32", (wk_size,), name="work_tensor_ub", scope=UB
                )
                # 如果work_tensor有索引，需要写成 work_tensor[index:]
                self.tik_instance.vec_rec_high_preci(
                    mask_len,
                    vec_rec_ub[0:],
                    vec_ub[0:],
                    work_tensor_ub[0:],
                    repeat_times,
                    dst_rep_stride,
                    src_rep_stride,
                )

            mask_len = vec_len - repeat_times * mask_len
            if mask_len > 0:
                wk_size = work_size * ((mask_len + block_len - 1) // block_len) * block_len
                work_tensor_ub2 = self.tik_instance.Tensor(
                    "float32", (wk_size,), name="work_tensor_ub2", scope=UB
                )
                self.tik_instance.vec_rec_high_preci(
                    mask_len,
                    vec_rec_ub[repeat_times * 128:],
                    vec_ub[repeat_times * 128:],
                    work_tensor_ub2[0:],
                    1,
                    0,
                    0,
                )
        return vec_rec_ub

    def row_sum_cube_impl(self, matrix_l1_K1MK0_ed, right_all_one_matrix_l1, rowsum_ub, m, k, precision_type):
        """用cube实现矩阵行和：右乘一个shape=(n,1)全一矩阵
        :param matrix_l1_K1MK0_ed: input tensor with shape (K1, M, K0)
        :param right_all_one_matrix_l1: input tensor with shape (K, 16)
        :param rowsum_ub: output tensor stores the row sum of input tensor
        :param m: actual tensor height
        :param k: actual tensor width
        :return: row sum of the output tensor
        """
        K1, M, K0 = matrix_l1_K1MK0_ed.shape
        # 调用matmul实现rowsum，结果shape=(m, 16)，取每行的第一个数
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            row_sum_ub_N1MN0 = self.matmul_compute(matrix_l1_K1MK0_ed, right_all_one_matrix_l1, m, k, 16,
                                                   N1MN0_to_MN=False, precision_type=precision_type)
            row_sum_ub_MN_ed = row_sum_ub_N1MN0.reshape((M, 16))
            if precision_type == FP32:
                for idx in range(0, m):
                    cur_row_sum = self.tik_instance.Scalar(FP32, init_value=row_sum_ub_MN_ed[idx, 0])
                    rowsum_ub[idx].set_as(cur_row_sum)
            else:
                # row_sum_ub_MN_ed 先转置，然后取一行, 替换原来按行操作: lij_ub[i].set_as(row_sum_ub_MN_ed[i, 0])
                row_sum_ub_trans = self.tik_instance.Tensor(FP16, (16, M), name="row_sum_ub_trans", scope=UB)
                row_sum_ub_trans = self.transpose_matrix(row_sum_ub_MN_ed, row_sum_ub_trans, M, True)
                self.cont_data_mv_1_bust(dst=rowsum_ub, src=row_sum_ub_trans, burst=M // 16)

        return rowsum_ub

    def matmul_compute(self, A_l1, B_l1, m, k, n, N1MN0_to_MN=True, precision_type=FP16):
        """calculate matrix multiplication A_l1 * B_l1, and move the result to C_ub,
        then rearrange C_ub
        :param A_l1: input tensor in L1 with shape of (K1, M, K0)
        :param B_l1: input tensor in L1 with shape of (K1, N, K0)
        :param m: the actual number of rows of A_l1
        :param k: the actual number of cols of A_l1
        :param n: the actual number of cols of B_l1
        :param N1MN0_to_MN: Whether reorder the result tensor.
        :return: C_ub with tensor with shape of (M, N) if N1MN0_to_MN else (N1, M, N0)
        """
        M = self.up_align_to_K0(m)
        N = self.up_align_to_K0(n)
        C_ub = self.tik_instance.Tensor(precision_type, (N // 16, M, 16), name="C_ub", scope=UB)
        try:
            dtype_size = DTYPE_SIZE[FP32]
        except KeyError:
            raise ValueError("The argument 'dtype' is not valid.")
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # matmul
            C_l0c = self.tik_instance.Tensor(
                FP32, (N // 16, M, 16), scope=L0C, name="C_l0c"
            )  # n1mn0 (n0=16)
            self.tik_instance.matmul(C_l0c, A_l1, B_l1, m, k, n)
            # L0C -> ub, fp32 -> fp16 (tensor_mov可做随路转换)
            self.tik_instance.tensor_mov(C_ub, C_l0c, "m", 1, M * N * dtype_size // 1024, 0, 0)
        if N1MN0_to_MN:
            return self.N1MN0_TO_MN(C_ub)
        return C_ub

    def move_vector_from_gm_to_ub(self, dst_tensor, src_tensor, gm_offset, vec_len):
        """load the vector from gm to ub
        :param dst_tensor:
        :param src_tensor:
        :param gm_offset:
        :return:
        """
        try:
            dtype_size = DTYPE_SIZE[src_tensor.dtype]
        except KeyError:
            raise ValueError("The argument 'src_tensor dtype' is not valid.")
        a_burst_num = 32 // dtype_size
        full_tik_blk_num, tail_num = divmod(vec_len, a_burst_num)
        with self.tik_instance.if_scope(full_tik_blk_num > 0):
            self.cont_data_mv_1_bust(dst=dst_tensor, src=src_tensor[gm_offset],
                                     burst=full_tik_blk_num)
        # 地址回退处理尾部数据
        with self.tik_instance.if_scope(tail_num > 0):
            offset = vec_len - a_burst_num
            last_blk_ub = self.tik_instance.Tensor(FP16, (a_burst_num,), name="last_blk_ub", scope=UB)
            self.cont_data_mv_1_bust(dst=last_blk_ub, src=src_tensor[gm_offset + offset], burst=1)
            with self.tik_instance.for_range(0, a_burst_num) as idx:  # offset非32bytes对齐, 无法用datamove
                dst_tensor[offset + idx].set_as(last_blk_ub[idx])

    def move_vector_from_ub_to_gm(self, dst_tensor, src_tensor, gm_offset, block_h):
        """write the vector back to gm
        :param dst_tensor:
        :param src_tensor:
        :param gm_offset:
        :param block_h:
        :return:
        """
        try:
            dtype_size = DTYPE_SIZE[src_tensor.dtype]
        except KeyError:
            raise ValueError("The argument 'src_tensor dtype' is not valid.")
        a_burst_num = 32 // dtype_size
        full_tik_blk_num = block_h // a_burst_num
        with self.tik_instance.if_scope(full_tik_blk_num > 0):
            self.cont_data_mv_1_bust(dst=dst_tensor[gm_offset], src=src_tensor,
                                     burst=full_tik_blk_num)
        tail_num = block_h % a_burst_num
        with self.tik_instance.if_scope(tail_num > 0):
            offset = block_h - a_burst_num
            tmp_ub = self.tik_instance.Tensor(FP16, (a_burst_num,), name="tmp_ub", scope=UB)
            with self.tik_instance.for_range(0, a_burst_num) as idx:
                tmp_ub[idx].set_as(src_tensor[offset + idx])
            self.cont_data_mv_1_bust(dst=dst_tensor[gm_offset + offset], src=tmp_ub, burst=1)

    def scale_compute_vector(self, Sij_ub, dim):
        """scale compute vector"""
        scale_value = dim ** -0.5
        scale = self.tik_instance.Scalar(dtype=FP16)
        scale.set_as(scale_value)
        self.tik_instance.h_mul(Sij_ub, Sij_ub, scale)
        return Sij_ub
