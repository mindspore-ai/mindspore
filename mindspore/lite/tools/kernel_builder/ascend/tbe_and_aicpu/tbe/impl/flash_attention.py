#!/usr/bin/python
# -*- coding: utf-8 -*-
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
"""ascend custom op: flash_attention by tik"""

import math
from collections import defaultdict
import te.platform as tbe_platform
from tbe import tik
from tbe.common.platform import get_soc_spec
from tbe.common.platform import set_current_compile_soc_info
from functools import partial

# set_current_compile_soc_info("Ascend910")
set_current_compile_soc_info("Ascend910ProB")

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
ENABLE_BROADCAST = True
INPUT_WITH_N1MN0 = False
OUTPUT_WITH_N1MN0 = False
AUTO_ROAD_CONVERSION = True
REMOVE_REPEAT_FRACTAL = True
FRONT_CONSTRUCT_ONE_MATAIX = True
SOFTMAX_WITH_NZ = True
ROWMAX_IMPL = True
DATA_WITH_NZ = True


class FlashAttention:
    """
    FlashAttention.
    """

    def __init__(self, q, k, v, disable_debug=True):
        self.tik_instance = tik.Tik(disable_debug=disable_debug)
        self.core_num = get_soc_spec(tbe_platform.CORE_NUM)
        self.M = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)

        self.cont_data_mv_1_bust = partial(self.tik_instance.data_move, sid=0, nburst=1,
                                           src_stride=0, dst_stride=0)

        self.update_sub_core_strategy = UPDATE_SUB_CORE_STRATEGY
        self.enable_broadcast = ENABLE_BROADCAST
        self.input_with_n1mn0 = INPUT_WITH_N1MN0
        self.output_with_n1mn0 = OUTPUT_WITH_N1MN0
        self.auto_road_conversion = AUTO_ROAD_CONVERSION
        self.remove_repeat_fractal = REMOVE_REPEAT_FRACTAL
        self.front_construct_one_matrix = FRONT_CONSTRUCT_ONE_MATAIX
        self.softmax_with_nz = SOFTMAX_WITH_NZ
        self.rowmax_impl = ROWMAX_IMPL
        self.data_with_nz = DATA_WITH_NZ

        if self.data_with_nz:
            # NZ
            self.q_shape = q["shape"]
            self.k_shape = k["shape"]
            self.v_shape = v["shape"]
            _, N1, M1, M0, N0 = self.q_shape
            self.M1 = M1
            self.N1 = N1
            self.M0 = M0
            self.N0 = N0
            self.d = N1 * N0
            # ND
            self.q_ori_shape = q["ori_shape"]
            self.k_ori_shape = k["ori_shape"]

            self.B, self.Nq, self.actual_d = self.q_ori_shape
            self.N = self.k_ori_shape[1]
            self.O_shape = self.q_shape
        else:
            if isinstance(q, dict):
                self.q_shape = q["shape"]  # [B, Nq, d], B = batch_size * head_num
                self.k_shape = k["shape"]  # [B, N, d]
                self.v_shape = v["shape"]  # [B, N, dv]
            else:
                self.q_shape = q.shape  # [B, Nq, d], B = batch_size * head_num
                self.k_shape = k.shape  # [B, N, d]
                self.v_shape = v.shape  # [B, N, dv]

            self.actual_d = self.q_shape[-1]
            self.B, self.Nq, self.d = self.q_shape
            self.N = self.k_shape[1]
            self.O_shape = [self.B, self.Nq, self.d]

        self.l_shape = [self.B, self.Nq]
        self.m_shape = [self.B, self.Nq]

        self.dtype = "float16"
        self.K0 = 16

    def tiling_for_wukonghuahua(self):
        """
        tiling for model.
        """
        if self.N <= 77:  # [77, 64]
            # cross-attention or self-attention of (64, 64, 160)
            self.Bc = self.N
            self.Tc = self.N // self.Bc
            if self.d <= 80:  # [40, 80]
                # 512 * 80 * 6 // 1024 = 240KB
                self.Br = min(self.Nq, 512)
                self.Tr = self.Nq // self.Br
            else:
                # dv = 160， 256 * 160 * 6 // 1024 = 240KB
                self.Br = min(self.Nq, 256)
                self.Tr = self.Nq // self.Br
        else:
            # self-attention
            if self.N == 256:
                self.Bc = 256
                self.Tc = 1
                # 128 * 256 * 6 // 1024 = 192KB
                self.Br = 128
                self.Tr = self.Nq // self.Br
            else:
                self.Bc = 512
                self.Tc = self.N // self.Bc
                # 64 * 512 * 6 // 1024 = 192KB
                self.Br = 64
                self.Tr = self.Nq // self.Br
        # for double buffer.
        self.Br = self.Br // 2
        self.Tr = self.Tr * 2

        self.last_Br = self.Br
        self.last_Bc = self.Bc

        self.tik_instance.tikdb.debug_print('"self.Bc:", self.Bc')
        self.tik_instance.tikdb.debug_print('"self.Br:", self.Br')
        self.tik_instance.tikdb.debug_print('"self.Tr:", self.Tr')
        self.tik_instance.tikdb.debug_print('"self.Tc:", self.Tc')

    def tiling_for_meitu(self):
        """
        tiling for model.
        """
        if self.d == 512:
            self.Bc = 32
            self.Br = 32
        elif self.N == 14144 or self.N == 10560:
            self.Bc = 256
            self.Br = 128 if self.auto_road_conversion else 64
        elif self.N < 512:
            self.Bc = 128
            self.Br = 128
        else:
            self.Bc = 512
            self.Br = 64 if self.auto_road_conversion else 32
        self.Tc = math.ceil(self.N / self.Bc)
        self.Tr = math.ceil(self.Nq / self.Br)
        self.last_Bc = self.Bc if self.N % self.Bc == 0 else self.N % self.Bc
        self.last_Br = self.Br if self.Nq % self.Br == 0 else self.Nq % self.Br

        self.tik_instance.tikdb.debug_print('"self.Bc:", self.Bc')
        self.tik_instance.tikdb.debug_print('"self.Br:", self.Br')
        self.tik_instance.tikdb.debug_print('"self.Tr:", self.Tr')
        self.tik_instance.tikdb.debug_print('"self.Tc:", self.Tc')
        self.tik_instance.tikdb.debug_print('"self.last_Br:", self.last_Br')
        self.tik_instance.tikdb.debug_print('"self.last_Bc:", self.last_Bc')

    def init(self):
        self.tiling_for_meitu()
        self.init_gm()
        if self.front_construct_one_matrix:
            self.right_all_one_matrix_l1 = self.construct_all_one_matrix(self.Bc)

    def init_gm(self):
        # define inputs
        self.q_gm = self.tik_instance.Tensor(FP16, self.q_shape, name="q_gm", scope=GM)
        self.k_gm = self.tik_instance.Tensor(FP16, self.k_shape, name="k_gm", scope=GM)
        self.v_gm = self.tik_instance.Tensor(FP16, self.v_shape, name="v_gm", scope=GM)

        # define output and intermediate res
        self.O_gm = self.tik_instance.Tensor(FP16, self.O_shape, name="O_gm", scope=GM)
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
        avg_batch_num_per_core, remain_batch = divmod(self.B, self.core_num)
        for core_idx in range(self.core_num):
            cur_core_batch_num = avg_batch_num_per_core
            if core_idx < remain_batch:
                cur_core_batch_num += 1

            core_batch_info[2 * core_idx] = core_batch_start
            core_batch_info[2 * core_idx + 1] = cur_core_batch_num
            core_batch_start += cur_core_batch_num

        return core_batch_info

    def get_each_core_task_info(self):
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
                batch_idx, tr_idx = task_idx_to_batch_tr_idx[task_idx]  # batch_idx: 0~16, tr_idx: 0~128
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

    def get_gm_offset(self, batch_start, batch_idx, h, w, block_h, block_idx):
        if self.data_with_nz:
            gm_offset = (batch_start + batch_idx) * h * w + block_idx * block_h * self.N0
        else:
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
            with self.tik_instance.for_range(0, 16) as idx:  # offset非32bytes对齐，无法用datamove
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

    def MK_TO_K1MK0(self, mk_input_tensor, workspace_tensor=None):
        """change data shape from (M, K) to (K1, M, K0), K1 = K // K0, the effect is equant to:
        new_tensor =  np.stack(np.hsplit(mk_input_tensor, K1), axis=0)

        :param mk_input_tensor: input tensor in GM with shape: (M, K)
        :param workspace_tensor: workspace tensor with shape: (K1, M, K0)
        tensor will be changed, otherwise the new data will be copied to the workspace tensor,
        and input tensor will stay unchanged.
        :return:
        """
        dtype = mk_input_tensor.dtype
        m, k = mk_input_tensor.shape
        K0 = 16
        K1 = k // K0
        M = self.up_align_to_K0(m)
        if workspace_tensor is not None:
            with self.tik_instance.for_range(0, K1) as i:
                self.tik_instance.data_move(
                    workspace_tensor[i * M * K0:],
                    mk_input_tensor[i * K0:],
                    0,
                    M,
                    K0 * DTYPE_SIZE[dtype] // 32,
                    (K1 - 1) * K0 * DTYPE_SIZE[dtype] // 32,
                    0,
                )
            return workspace_tensor.reshape((K1, M, K0))
        else:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                tmp_ub = self.tik_instance.Tensor(dtype, (K1, M, K0), name="tmp_ub", scope=UB)
                with self.tik_instance.for_range(0, K1) as i:
                    self.tik_instance.data_move(
                        tmp_ub[i * M * K0:],
                        mk_input_tensor[i * K0:],
                        0,
                        M,
                        K0 * DTYPE_SIZE[dtype] // 32,
                        (K1 - 1) * K0 * DTYPE_SIZE[dtype] // 32,
                        0,
                    )
                self.tik_instance.data_move(
                    mk_input_tensor, tmp_ub, 0, 1, K1 * M * K0 * DTYPE_SIZE[dtype] // 32, 0, 0
                )
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

    def construct_all_one_matrix(self, K):
        right_all_one_matrix_ub = self.tik_instance.Tensor(
            FP16, (K, 16), name="right_all_one_matrix_ub", scope=UB
        )
        self.tik_instance.h_duplicate(right_all_one_matrix_ub, 1.0)
        right_all_one_matrix_l1 = self.tik_instance.Tensor(
            FP16, (K, 16), name="right_all_one_matrix_l1", scope=L1
        )
        self.tik_instance.data_move(right_all_one_matrix_l1,
                                    right_all_one_matrix_ub,
                                    0, 1, K, 0, 0)
        return right_all_one_matrix_l1

    def KN_TO_K1NK0(self, kn_input_tensor, workspace_tensor=None):
        """change data shape from (K,N) to (K1, N, K0), K1 = K // K0, the effect is equant to:
        new_tensor =  np.reshape(kn_input_tensor, newshape=(K1, K0, N)).swapaxes(1, 2)

        :param kn_input_tensor: input tensor with shape: (K, N)
        :param workspace_tensor: workspace tensor with shape: (K1, N, K0)
        tensor will be changed, otherwise the new data will be copied to the workspace tensor,
        and input tensor will stay unchanged.
        :return:
        """
        dtype = kn_input_tensor.dtype
        k, n = kn_input_tensor.shape
        K0 = 16
        K1 = k // K0
        N = n
        with self.tik_instance.for_range(0, K1) as index:
            k1nk0_ub = self.tik_instance.Tensor(dtype, (N, K0), UB, "k1nk0_ub")
            src_ub = self.tik_instance.Tensor(dtype, (K0, N), UB, "src_ub")
            burst_len = K0 * N * DTYPE_SIZE[dtype] // 32
            self.tik_instance.data_move(
                src_ub, kn_input_tensor[index * K0 * N], 0, 1, burst_len, 0, 0
            )
            k1nk0_ub = self.transpose_matrix(src_ub, k1nk0_ub, N)
            if workspace_tensor is None:
                self.tik_instance.data_move(
                    kn_input_tensor[index * K0 * N], k1nk0_ub, 0, 1, burst_len, 0, 0
                )
            else:
                self.tik_instance.data_move(
                    workspace_tensor[index * K0 * N], k1nk0_ub, 0, 1, burst_len, 0, 0
                )
        if workspace_tensor is None:
            return kn_input_tensor.reshape((K1, N, K0))
        else:
            return workspace_tensor.reshape((K1, N, K0))

    def KN_TO_K1NK0_V2(self, kn_input_tensor, workspace_tensor):
        """change data shape from (K,N) to (K1, N, K0), K1 = K // K0, the effect is equant to:
        new_tensor =  np.reshape(kn_input_tensor, newshape=(K1, K0, N)).swapaxes(1, 2)

        :param kn_input_tensor: input tensor with shape: (K, N)
        :param workspace_tensor: workspace tensor with shape: (K1, N, K0)
        tensor will be changed, otherwise the new data will be copied to the workspace tensor,
        and input tensor will stay unchanged.
        :return:
        """
        dtype = kn_input_tensor.dtype
        k, n = kn_input_tensor.shape
        K0 = 16
        K1 = k // K0
        N = n
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            workspace_tensor_tmp = self.tik_instance.Tensor(dtype, (k, n), UB, "workspace_tensor_tmp")
            # k1k0n -> k0k1n
            with self.tik_instance.for_range(0, K0) as index:
                self.tik_instance.data_move(workspace_tensor_tmp[index * K1 * N], kn_input_tensor[index * N],
                                            0, K1, N // 16, (K0 - 1) * N // 16, 0)
            # k0(k1n) -> k1nk0
            kn_input_tensor = self.transpose_matrix(workspace_tensor_tmp, kn_input_tensor, K1 * N)
        # ub -> l1
        self.tik_instance.data_move(workspace_tensor, kn_input_tensor, 0, 1, k * n // 16, 0, 0)
        return workspace_tensor.reshape((K1, N, K0))

    def broadcast(self, vec_ub, shape):
        ''' broadcast a vector to a matrix
        :param vec: a tensor in UB with shape of (M,), and dtype is float16
        :param shape: the target shape, a tuple with value (M, N)，M and N are integer multiples of 16
        :return: a tensor in UB with shape of (M, N)
        '''
        M, N = shape
        dst_ub = self.tik_instance.Tensor(FP16, shape, name="dst_ub", scope=UB)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # (M,) -> (2, M) -> (4, M) -> (8, M) -> (16, M)
            tmp_ub1 = self.tik_instance.Tensor(FP16, (16, M), name="tmp_ub1", scope=UB)
            self.tik_instance.data_move(tmp_ub1, vec_ub, 0, 1, M // 16, 0, 0)
            times = self.tik_instance.Scalar("int32", name="times", init_value=1)
            with self.tik_instance.for_range(begint=0, endt=16) as idx:
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
            with self.tik_instance.for_range(begint=0, endt=N) as idx:
                offset = times * 16
                with self.tik_instance.if_scope(offset * 2 <= N):
                    burst = offset // 16
                    src_stride = N // 16 - burst
                    dst_stride = N // 16 - burst
                    self.tik_instance.data_move(dst_ub[offset], dst_ub, 0, M, burst, src_stride, dst_stride)
                with self.tik_instance.else_scope():
                    burst = (N - offset) // 16
                    src_stride = N // 16 - burst
                    dst_stride = N // 16 - burst
                    with self.tik_instance.if_scope(burst > 0):
                        self.tik_instance.data_move(dst_ub[offset], dst_ub, 0, M, burst, src_stride, dst_stride)
                    self.tik_instance.tik_break()
                times.set_as(times * 2)
        return dst_ub

    def N1MN0_TO_MN(self, N1MN0_input):
        """change data shape from (N1, M, N0) to (M, N), N0=16, N = N1 * K0, the effect is equant to:
        N1MN0_input = np.concatenate(list(map(np.squeeze, np.split(N1MN0_input, N1))), axis=1)

        :param N1MN0_input: input tensor with shape (N, M, N0) in GM or L1.
        :return:
        """
        dtype = N1MN0_input.dtype
        N1, M, N0 = N1MN0_input.shape

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            tmp_ub = self.tik_instance.Tensor(dtype, (M, N1 * N0), name="tmp_ub", scope=UB)
            # data_move (n1,m,n0) --> (m,n)
            with self.tik_instance.for_range(0, N1) as i:
                self.tik_instance.data_move(
                    tmp_ub[i * N0:],
                    N1MN0_input[i * M * N0:],
                    0,
                    M,
                    N0 * DTYPE_SIZE[dtype] // 32,
                    0,
                    (N1 - 1) * N0 * DTYPE_SIZE[dtype] // 32,
                )
            # data_move out
            self.tik_instance.data_move(
                N1MN0_input, tmp_ub, 0, 1, M * N1 * N0 * DTYPE_SIZE[dtype] // 32, 0, 0
            )
        return N1MN0_input.reshape((M, N1 * N0))

    def matmul_compute_with_road_conversion(self, A_l1, B_l1, m, k, n, reorder_res=True, precision_type=FP16):
        """calculate matrix multiplication A_l1 * B_l1, and move the result to C_ub,
        then rearrange C_ub
        :param A_l1: input tensor in L1 with shape of (K1, M, K0)
        :param B_l1: input tensor in L1 with shape of (K1, N, K0)
        :param m: the actual number of rows of A_l1
        :param k: the actual number of cols of A_l1
        :param n: the actual number of cols of B_l1
        :param reorder_res: Whether reorder the result tensor.
        :return: C_ub with tensor with shape of (M, N) if reorder_res else (N1, M, N0)
        """
        M = self.up_align_to_K0(m)
        N = self.up_align_to_K0(n)
        C_ub = self.tik_instance.Tensor(precision_type, (N // 16, M, 16), name="C_ub", scope=UB)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # matmul
            C_l0c = self.tik_instance.Tensor(
                FP32, (N // 16, M, 16), scope=L0C, name="C_l0c"
            )  # n1mn0 (n0=16)
            self.tik_instance.matmul(C_l0c, A_l1, B_l1, m, k, n)
            self.tik_instance.tensor_mov(C_ub, C_l0c, "m", 1, M * N * DTYPE_SIZE[FP32] // 1024, 0, 0)
        if reorder_res:
            return self.N1MN0_TO_MN(C_ub)
        else:
            return C_ub

    def matmul_compute(self, A_l1, B_l1, C_ub, m, k, n, k0=16, reorder_res=True):
        """calculate matrix multiplication A_l1 * B_l1, and move the result to C_ub,
        then rearrange C_ub
        :param A_l1: input tensor in L1 with shape of (k_alig // 16, m_alig, 16)
        :param B_l1: input tensor in L1 with shape of (k_alig // 16, n_alig, 16)
        :param C_ub: workspace tensor in UB with shape of (n_alig // 16, m_alig, 16)
        :param m: the actual number of rows of A_l1
        :param k: the actual number of cols of A_l1
        :param n: the actual number of cols of B_l1
        :param k0: matrix fractal param
        :return: C_ub with tensor with shape of (m_alig, n_alig)
        """
        m_alig = self.up_align_to_K0(m)
        n_alig = self.up_align_to_K0(n)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # matmul
            C_l0c = self.tik_instance.Tensor(
                FP32, (n_alig // k0, m_alig, k0), scope=L0C, name="C_l0c"
            )  # n1mn0 (n0=16)
            self.tik_instance.matmul(C_l0c, A_l1, B_l1, m, k, n)
            t_num = 2
            if m_alig * n_alig == 256:
                t_num = 1
            with self.tik_instance.for_range(0, t_num, thread_num=t_num) as t_id:
                tmp_C_ub_fp32 = self.tik_instance.Tensor(
                    FP32, (m_alig * n_alig // t_num,), name="tmp_C_ub_fp32", scope=UB)
                tmp_C_ub_fp16 = self.tik_instance.Tensor(
                    FP16, (m_alig * n_alig // t_num,), name="tmp_C_ub_fp16", scope=UB)
                offset = t_id * (m_alig * n_alig // t_num)
                self.tik_instance.data_move(tmp_C_ub_fp32,
                                            C_l0c[offset],
                                            0,
                                            1,
                                            m_alig * n_alig // t_num * DTYPE_SIZE[FP32] // 1024,
                                            0,
                                            0)
                self.tik_instance.h_cast(tmp_C_ub_fp16, tmp_C_ub_fp32, "none")
                self.tik_instance.data_move(C_ub[offset],
                                            tmp_C_ub_fp16,
                                            0,
                                            1,
                                            m_alig * n_alig // t_num * DTYPE_SIZE[FP16] // 32,
                                            0,
                                            0)
        if reorder_res:
            return self.N1MN0_TO_MN(C_ub)
        else:
            return C_ub

    def scale_compute_vector(self, Sij_ub, dim):
        scale_value = dim ** -0.5
        scale = self.tik_instance.Scalar(dtype=FP16)
        scale.set_as(scale_value)
        self.tik_instance.h_mul(Sij_ub, Sij_ub, scale)
        return Sij_ub

    def softmax_compute(self, Sij_ub, mij_ub, lij_ub, m, n):
        """
        Sij_ub shape M x N
        使用Sij空间返回Pij 提高UB利用率
        """
        self.tik_instance.h_reduce_max(mij_ub, Sij_ub[0:m, 0:n], 1)
        if self.enable_broadcast:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                m_aligned = self.up_align_to_K0(m)
                n_aligned = self.up_align_to_K0(n)
                broadcast_mij_ub = self.broadcast(mij_ub, (m_aligned, n_aligned))
                self.tik_instance.h_sub(Sij_ub, Sij_ub, broadcast_mij_ub)
        else:
            with self.tik_instance.for_range(0, m) as i:
                src_scalar = self.tik_instance.Scalar(init_value=mij_ub[i], dtype=FP16)
                self.tik_instance.h_sub(Sij_ub[i, :], Sij_ub[i, :], src_scalar)
        # exp
        self.tik_instance.h_exp(Sij_ub, Sij_ub)

        # cube impl rowsum
        m_alig = self.up_align_to_K0(m)
        n_alig = self.up_align_to_K0(n)
        Sij_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (n_alig // 16, m_alig, 16), name="Sij_l1_K1MK0_ed", scope=L1)
        Sij_l1_K1MK0_ed = self.MK_TO_K1MK0(Sij_ub, Sij_l1_K1MK0_ed)
        lij_ub = self.row_sum_cube_impl(Sij_l1_K1MK0_ed, lij_ub, m, n)

        return Sij_ub, mij_ub, lij_ub

    def softmax_compute_with_fractal(self, Sij_l1_K1MK0_ed, Sij_ub, mij_ub, lij_ub, m, n):
        """
        Sij_ub shape M x N
        """
        self.tik_instance.h_reduce_max(mij_ub, Sij_ub[0:m, 0:n], 1)
        if self.enable_broadcast:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                m_aligned = self.up_align_to_K0(m)
                n_aligned = self.up_align_to_K0(n)
                broadcast_mij_ub = self.broadcast(mij_ub, (m_aligned, n_aligned))
                self.tik_instance.h_sub(Sij_ub, Sij_ub, broadcast_mij_ub)
        else:
            # Sij - mij
            with self.tik_instance.for_range(0, m) as i:
                src_scalar = self.tik_instance.Scalar(init_value=mij_ub[i], dtype=FP16)
                self.tik_instance.h_sub(Sij_ub[i, :], Sij_ub[i, :], src_scalar)
        # exp
        self.tik_instance.h_exp(Sij_ub, Sij_ub)

        # cube impl rowsum
        Sij_l1_K1MK0_ed = self.MK_TO_K1MK0(Sij_ub, Sij_l1_K1MK0_ed)
        lij_ub = self.row_sum_cube_impl(Sij_l1_K1MK0_ed, lij_ub, m, n)

        return Sij_l1_K1MK0_ed, mij_ub, lij_ub

    def softmax_compute_with_nz(self, Sij_l1_K1MK0_ed, Sij_ub, mij_ub, lij_ub, m, n):
        """Refer to Algorithm 2 line12
        Calculate softmax.
        :param Sij_ub: with shape M x N or N1 x M x N0, 使用Sij空间返回Pij 提高UB利用率
        :param mij_ub:
        :param lij_ub:
        :param m:
        :param n:
        :return:
        """
        m_aligned = self.up_align_to_K0(m)
        n_aligned = self.up_align_to_K0(n)

        if len(Sij_ub.shape) == 2:  # [M, N]
            self.tik_instance.h_reduce_max(mij_ub, Sij_ub[:, 0:n], 1)
            # Sij - mij
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                broadcast_mij_ub = self.broadcast(mij_ub, (m_aligned, n_aligned))
                self.tik_instance.h_sub(Sij_ub, Sij_ub, broadcast_mij_ub)
                self.tik_instance.h_exp(Sij_ub, Sij_ub)
        else:  # [N1, M, N0]
            n0 = 16
            n1 = n // 16
            if self.rowmax_impl:
                cur_block_rowmax_ub = self.tik_instance.Tensor(FP16, (1, m, 16), name="cur_block_rowmax_ub", scope=UB)
                self.tik_instance.data_move(cur_block_rowmax_ub, Sij_ub[0], 0, m * 16 // 16, 1, 0, 0)
                with self.tik_instance.for_range(1, n1) as idx:
                    self.tik_instance.h_max(cur_block_rowmax_ub, Sij_ub[idx, :, :], cur_block_rowmax_ub)
                cur_block_rowmax_ub = cur_block_rowmax_ub.reshape((m, 16))
                self.tik_instance.h_reduce_max(mij_ub, cur_block_rowmax_ub, 1)
            else:
                # calc rowmax of Sij
                self.tik_instance.h_duplicate(mij_ub, FP16_MIN_VAL)
                with self.tik_instance.for_range(0, n1) as idx:
                    cur_block_rowmax_ub = self.tik_instance.Tensor(FP16, (1, m), name="cur_block_rowmax_ub", scope=UB)
                    self.tik_instance.h_reduce_max(cur_block_rowmax_ub, Sij_ub[idx, :, :], 2)
                    cur_block_rowmax_ub = cur_block_rowmax_ub.reshape((m,))
                    self.tik_instance.h_max(mij_ub, mij_ub, cur_block_rowmax_ub)
            # Sij - mij
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                broadcast_mij_ub = self.broadcast(mij_ub, (m, n0))
                broadcast_mij_ub = broadcast_mij_ub.reshape((1, m, n0))
                for idx in range(n1):
                    self.tik_instance.h_sub(Sij_ub[idx, :, :], Sij_ub[idx, :, :], broadcast_mij_ub)

        self.tik_instance.h_exp(Sij_ub, Sij_ub)

        # cube impl rowsum
        if len(Sij_ub.shape) == 2:  # [M, N]
            # [M, N] -> [N1, M, N0]
            Sij_l1_K1MK0_ed = self.MK_TO_K1MK0(Sij_ub, Sij_l1_K1MK0_ed)
        else:  # [N1, M, N0]
            self.cont_data_mv_1_bust(dst=Sij_l1_K1MK0_ed, src=Sij_ub, burst=m * n // 16)
        Sij_row_sum_ub = self.row_sum_cube_impl(Sij_l1_K1MK0_ed, lij_ub, m, n)

        return Sij_l1_K1MK0_ed, mij_ub, Sij_row_sum_ub

    def update_m_l(self, mi_old_ub, mij_ub, li_old_ub, lij_ub, vec_len):
        mi_new_ub = self.tik_instance.Tensor(FP16, (vec_len,), name="mi_new_ub", scope=UB)
        li_new_ub = self.tik_instance.Tensor(FP16, (vec_len,), name="li_new_ub", scope=UB)
        # 1 mi_new = max(mi, mij)
        self.tik_instance.h_max(mi_new_ub, mi_old_ub, mij_ub)
        # self.tik_instance.tikdb.debug_print("m_i_new_ub")

        # 2 li_new = exp(mi-mi_new)*li + exp(mij-mi_new)*lij
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
        # self.tik_instance.tikdb.debug_print("li_new_ub")
        return mi_new_ub, li_new_ub

    def clac_new_Oi(
            self,
            Oi_ub,
            exp_mi_sub_mi_new,
            Pij_Vj_ub,
            exp_mij_sub_mi_new,
            li_new_rec_ub,
            li_ub,
            block_h,
            block_w,
    ):
        """Oi_new = (l_i_old * exp(m_i_old - m_i_new) * Oi_old + exp(m_ij - m_i_new) * Pij*Vj) / l_i_new
        :param Oi_ub:
        :param exp_mi_sub_mi_new:
        :param Pij_Vj_ub: (q_blk_h_aligned, self.d)
        :param exp_mij_sub_mi_new:
        :param li_new_rec_ub:
        :param li_ub:
        :param block_h:
        :param block_w:
        :return:
        """
        diag_exp_Oi_ub = self.diag_exp_Oi(li_ub, exp_mi_sub_mi_new, Oi_ub, block_h, block_w)

        exp_Pij_Vj_ub = self.exp_Pij_Vj(exp_mij_sub_mi_new, Pij_Vj_ub, block_h, block_w)

        # (diag(li)_exp_Oi + exp_P_V)
        # rshape: (block_h, block_w)
        sum_diag_exp_Oi_and_exp_Pij_Vj_ub = diag_exp_Oi_ub
        self.tik_instance.h_add(
            sum_diag_exp_Oi_and_exp_Pij_Vj_ub,
            sum_diag_exp_Oi_and_exp_Pij_Vj_ub,
            exp_Pij_Vj_ub
        )

        if self.enable_broadcast:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                block_h_aligned = self.up_align_to_K0((block_h))
                broadcast_li_new_rec_ub = self.broadcast(li_new_rec_ub, (block_h_aligned, self.d))
                self.tik_instance.h_mul(
                    sum_diag_exp_Oi_and_exp_Pij_Vj_ub,
                    sum_diag_exp_Oi_and_exp_Pij_Vj_ub,
                    broadcast_li_new_rec_ub
                )
        else:
            # (diag_li_exp_mi_sub_mi_new * Oi + exp_mij_sub_mi_new * Pij_Vj_ub) / l_i_new
            # rshape: (block_h, block_w)
            with self.tik_instance.for_range(begint=0, endt=block_h) as i:
                src_scalar = self.tik_instance.Scalar(init_value=li_new_rec_ub[i], dtype=FP16)
                self.tik_instance.h_mul(
                    sum_diag_exp_Oi_and_exp_Pij_Vj_ub[i, :],
                    src_scalar,
                    sum_diag_exp_Oi_and_exp_Pij_Vj_ub[i, :],
                )

        return sum_diag_exp_Oi_and_exp_Pij_Vj_ub

    def diag_exp_Oi(self, li_ub, exp_mi_sub_mi_new, Oi_ub, block_h, block_w):
        # diag(li) exp(mi-mi_new)
        self.tik_instance.h_mul(exp_mi_sub_mi_new, exp_mi_sub_mi_new, li_ub)
        diag_exp = exp_mi_sub_mi_new
        # op:   diag_exp        Oi
        # shape (block_h,)      (block_h, block_w)
        if self.enable_broadcast:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                block_h_aligned = self.up_align_to_K0(block_h)
                broadcast_diag_exp = self.broadcast(diag_exp, (block_h_aligned, self.d))
                self.tik_instance.h_mul(Oi_ub, Oi_ub, broadcast_diag_exp)
        else:
            with self.tik_instance.for_range(begint=0, endt=block_h) as i:
                src_scalar = self.tik_instance.Scalar(init_value=diag_exp[i], dtype=FP16)
                self.tik_instance.h_mul(Oi_ub[i, :], Oi_ub[i, :], src_scalar)
        return Oi_ub

    def exp_Pij_Vj(self, exp_mij_sub_mi_new, Pij_Vj_ub, block_h, block_w):
        # op:       exp_mij_sub_mi_new Pij_Vj_ub
        # shape:    (block_h,)        (block_h, block_w)
        # rshape:   (block_h, block_w)
        if self.enable_broadcast:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                block_h_aligned = self.up_align_to_K0(block_h)
                broadcast_exp_mij_sub_mi_new = self.broadcast(exp_mij_sub_mi_new, (block_h_aligned, self.d))
                self.tik_instance.h_mul(Pij_Vj_ub, Pij_Vj_ub, broadcast_exp_mij_sub_mi_new)
        else:
            with self.tik_instance.for_range(begint=0, endt=block_h) as i:
                src_scalar = self.tik_instance.Scalar(init_value=exp_mij_sub_mi_new[i], dtype=FP16)
                self.tik_instance.h_mul(Pij_Vj_ub[i, :], src_scalar, Pij_Vj_ub[i, :])
        return Pij_Vj_ub

    def calc_vec_rec(self, li_new_ub, block_h):
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
                # vec_rec_high_preci(mask, dst, src, work_tensor, repeat_times, dst_rep_stride, src_rep_stride)
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

    def update_o_m_l_n1mn0(self, Pij_Vj_ub, mij_ub, lij_ub, batch_start, batch_idx, kv_blk_idx, q_blk_idx, block_h):
        """ load o m l from gm and update them in ub, then write them back to gm
        :param Pij_Vj_ub: input tensor with shape of (q_blk_h_aligned, self.d)
        :param mij_ub: input tensor with shape of (Br)
        :param lij_ub: input tensor with shape of (Br)
        :param batch_start:
        :param batch_idx:
        :param kv_blk_idx:
        :param q_blk_idx:
        :param block_h:
        :return: None
        """
        n1 = self.d // 16
        vec_gm_offset = (batch_start + batch_idx) * self.Nq + q_blk_idx * self.Br
        o_gm_offset = self.get_gm_offset(
            batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx
        )
        h_alig = self.up_align_to_K0(block_h)
        # mi_new = mij, li_new=lij. Oi <- Pij*Vj/Li_new
        with self.tik_instance.if_scope(kv_blk_idx == 0):
            self.move_vector_from_ub_to_gm(self.l_gm, lij_ub, vec_gm_offset, block_h)
            self.move_vector_from_ub_to_gm(self.m_gm, mij_ub, vec_gm_offset, block_h)
            li_new_rec_ub = self.calc_vec_rec(lij_ub, block_h)
            if self.enable_broadcast:
                Pij_Vj_ub = self.vector_matrix_mul(li_new_rec_ub, Pij_Vj_ub, block_h, n1)
            else:
                with self.tik_instance.for_range(begint=0, endt=block_h) as i:
                    src_scalar = self.tik_instance.Scalar(init_value=li_new_rec_ub[i], dtype=FP16)
                    self.tik_instance.h_mul(Pij_Vj_ub[i, :], src_scalar, Pij_Vj_ub[i, :])
            self.tik_instance.data_move(self.O_gm[o_gm_offset], Pij_Vj_ub, 0, 1, block_h * self.d // 16, 0, 0)
        with self.tik_instance.else_scope():
            mi_ub = self.tik_instance.Tensor(FP16, (block_h,), name="mi_old_ub", scope=UB)
            li_ub = self.tik_instance.Tensor(FP16, (block_h,), name="li_ub", scope=UB)
            self.move_vector_from_gm_to_ub(mi_ub, self.m_gm, vec_gm_offset)
            self.move_vector_from_gm_to_ub(li_ub, self.l_gm, vec_gm_offset)

            mi_new_ub, li_new_ub = self.update_m_l(mi_ub, mij_ub, li_ub, lij_ub, block_h)
            exp_mi_sub_mi_new = mi_ub
            exp_mij_sub_mi_new = mij_ub
            Oi_ub = self.tik_instance.Tensor(FP16, (self.d // 16, h_alig, 16), scope=UB, name="Oi_ub")
            self.tik_instance.data_move(
                Oi_ub, self.O_gm[o_gm_offset], 0, 1, block_h * self.d // 16, 0, 0
            )

            li_new_rec_ub = self.calc_vec_rec(li_new_ub, block_h)
            Oi_new_ub = self.clac_new_Oi_n1mn0(
                Oi_ub,
                exp_mi_sub_mi_new,
                Pij_Vj_ub,
                exp_mij_sub_mi_new,
                li_new_rec_ub,
                li_ub,
                block_h,
                self.d,
            )

            self.tik_instance.data_move(self.O_gm[o_gm_offset], Oi_new_ub, 0, 1, block_h * self.d // 16, 0, 0)
            self.move_vector_from_ub_to_gm(self.l_gm, li_new_ub, vec_gm_offset, block_h)
            self.move_vector_from_ub_to_gm(self.m_gm, mi_new_ub, vec_gm_offset, block_h)

    def clac_new_Oi_n1mn0(self, Oi_ub, exp_mi_sub_mi_new, Pij_Vj_ub, exp_mij_sub_mi_new,
                          li_new_rec_ub, li_ub, block_h, block_w):
        """Oi_new = (l_i_old * exp(m_i_old - m_i_new) * Oi_old + exp(m_ij - m_i_new) * Pij*Vj) / l_i_new
        :param Oi_ub:
        :param exp_mi_sub_mi_new:
        :param Pij_Vj_ub: (q_blk_h_aligned, self.d)
        :param exp_mij_sub_mi_new:
        :param li_new_rec_ub:
        :param li_ub:
        :param block_h:
        :param block_w:
        :return:
        """
        n0 = 16
        n1 = block_w // 16
        diag_exp_Oi_ub = self.diag_exp_Oi_n1mn0(li_ub, exp_mi_sub_mi_new, Oi_ub, block_h, block_w)

        exp_Pij_Vj_ub = self.exp_Pij_Vj_n1mn0(exp_mij_sub_mi_new, Pij_Vj_ub, block_h, block_w)

        # (diag(li)_exp_Oi + exp_P_V)
        # rshape: (block_h, block_w)
        sum_diag_exp_Oi_and_exp_Pij_Vj_ub = diag_exp_Oi_ub
        self.tik_instance.h_add(
            sum_diag_exp_Oi_and_exp_Pij_Vj_ub,
            sum_diag_exp_Oi_and_exp_Pij_Vj_ub,
            exp_Pij_Vj_ub
        )

        if self.enable_broadcast:
            sum_diag_exp_Oi_and_exp_Pij_Vj_ub = self.vector_matrix_mul(li_new_rec_ub, sum_diag_exp_Oi_and_exp_Pij_Vj_ub,
                                                                       block_h, n1)
        else:
            # (diag_li_exp_mi_sub_mi_new * Oi + exp_mij_sub_mi_new * Pij_Vj_ub) / l_i_new
            # rshape: (block_h, block_w)
            with self.tik_instance.for_range(begint=0, endt=block_h) as i:
                src_scalar = self.tik_instance.Scalar(init_value=li_new_rec_ub[i], dtype=FP16)
                self.tik_instance.h_mul(
                    sum_diag_exp_Oi_and_exp_Pij_Vj_ub[i, :],
                    src_scalar,
                    sum_diag_exp_Oi_and_exp_Pij_Vj_ub[i, :],
                )
        return sum_diag_exp_Oi_and_exp_Pij_Vj_ub

    def diag_exp_Oi_n1mn0(self, li_ub, exp_mi_sub_mi_new, Oi_ub, block_h, block_w):
        # diag(li) exp(mi-mi_new)
        n0 = 16
        n1 = block_w // 16
        self.tik_instance.h_mul(exp_mi_sub_mi_new, exp_mi_sub_mi_new, li_ub)
        diag_exp = exp_mi_sub_mi_new
        # op:   diag_exp        Oi
        # shape (block_h,)      (block_h, block_w)
        Oi_ub = self.vector_matrix_mul(diag_exp, Oi_ub, block_h, n1)
        return Oi_ub

    def exp_Pij_Vj_n1mn0(self, exp_mij_sub_mi_new, Pij_Vj_ub, block_h, block_w):
        # op:       exp_mij_sub_mi_new Pij_Vj_ub
        # shape:    (block_h,)        (block_h, block_w)
        # rshape:   (block_h, block_w)
        n0 = 16
        n1 = block_w // 16
        Pij_Vj_ub = self.vector_matrix_mul(exp_mij_sub_mi_new, Pij_Vj_ub, block_h, n1)
        return Pij_Vj_ub

    def vector_matrix_mul(self, v, m, h, n1):
        broadcast_v = self.broadcast(v, (h, 16))
        broadcast_v = broadcast_v.reshape((1, h, 16))
        for nid in range(n1):
            self.tik_instance.h_mul(m[nid, :, :], broadcast_v, m[nid, :, :])
        return m

    def update_o_m_l(self, Pij_Vj_ub, mij_ub, lij_ub, batch_start, batch_idx, kv_blk_idx, q_blk_idx, block_h):
        """ load o m l from gm and update them in ub, then write them back to gm
        :param Pij_Vj_ub: input tensor with shape of (q_blk_h_aligned, self.d)
        :param mij_ub: input tensor with shape of (Br)
        :param lij_ub: input tensor with shape of (Br)
        :param batch_start:
        :param batch_idx:
        :param kv_blk_idx:
        :param q_blk_idx:
        :param block_h:
        :return: None
        """
        vec_gm_offset = (batch_start + batch_idx) * self.Nq + q_blk_idx * self.Br
        o_gm_offset = self.get_gm_offset(
            batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx
        )
        h_alig = self.up_align_to_K0(block_h)

        n1, m, n0 = Pij_Vj_ub.shape
        # mi_new = mij, li_new=lij. Oi <- Pij*Vj/Li_new
        with self.tik_instance.if_scope(kv_blk_idx == 0):
            self.move_vector_from_ub_to_gm(self.l_gm, lij_ub, vec_gm_offset, block_h)
            self.move_vector_from_ub_to_gm(self.m_gm, mij_ub, vec_gm_offset, block_h)
            li_new_rec_ub = self.calc_vec_rec(lij_ub, block_h)
            if self.data_with_nz:
                broadcast_li_new_rec_ub = self.broadcast(li_new_rec_ub, (m, n0))
                broadcast_li_new_rec_ub = broadcast_li_new_rec_ub.reshape((1, m, n0))
                with self.tik_instance.for_range(0, n1) as idx:
                    self.tik_instance.h_mul(Pij_Vj_ub[idx, :, :], Pij_Vj_ub[idx, :, :], broadcast_li_new_rec_ub)
                self.tik_instance.data_move(dst=self.O_gm[o_gm_offset], src=Pij_Vj_ub, sid=0,
                                            nburst=self.N1, burst=block_h * self.N0 // 16,
                                            src_stride=0, dst_stride=(self.Nq - h_alig) * self.N0 // 16)
            else:
                if self.enable_broadcast:
                    with self.tik_instance.new_stmt_scope(disable_sync=False):
                        broadcast_li_new_rec_ub = self.broadcast(li_new_rec_ub, (h_alig, self.d))
                        self.tik_instance.h_mul(Pij_Vj_ub, Pij_Vj_ub, broadcast_li_new_rec_ub)
                else:
                    with self.tik_instance.for_range(begint=0, endt=block_h) as i:
                        src_scalar = self.tik_instance.Scalar(init_value=li_new_rec_ub[i], dtype=FP16)
                        self.tik_instance.h_mul(Pij_Vj_ub[i, :], src_scalar, Pij_Vj_ub[i, :])
                self.tik_instance.data_move(self.O_gm[o_gm_offset], Pij_Vj_ub, 0, 1, block_h * self.d // 16, 0, 0)

        with self.tik_instance.else_scope():
            mi_ub = self.tik_instance.Tensor(FP16, (block_h,), name="mi_old_ub", scope=UB)
            li_ub = self.tik_instance.Tensor(FP16, (block_h,), name="li_ub", scope=UB)
            self.move_vector_from_gm_to_ub(mi_ub, self.m_gm, vec_gm_offset)
            self.move_vector_from_gm_to_ub(li_ub, self.l_gm, vec_gm_offset)

            mi_new_ub, li_new_ub = self.update_m_l(mi_ub, mij_ub, li_ub, lij_ub, block_h)
            self.move_vector_from_ub_to_gm(self.l_gm, li_new_ub, vec_gm_offset, block_h)
            self.move_vector_from_ub_to_gm(self.m_gm, mi_new_ub, vec_gm_offset, block_h)

            exp_mi_sub_mi_new = mi_ub
            exp_mij_sub_mi_new = mij_ub
            li_new_rec_ub = self.calc_vec_rec(li_new_ub, block_h)

            if self.data_with_nz:
                # scale1 <- li * exp(mi - mi_new) / li_new
                self.tik_instance.h_mul(li_ub, li_ub, exp_mi_sub_mi_new)
                self.tik_instance.h_mul(li_ub, li_ub, li_new_rec_ub)
                scale1 = li_ub

                # scale2 <- exp(mij - mi_new) / li_new
                self.tik_instance.h_mul(exp_mij_sub_mi_new, exp_mij_sub_mi_new, li_new_rec_ub)
                scale2 = exp_mij_sub_mi_new

                Oi_ub = self.tik_instance.Tensor(FP16, (n1, m, n0), name="Oi_ub", scope=UB)
                self.tik_instance.data_move(dst=Oi_ub, src=self.O_gm[o_gm_offset],
                                            sid=0, nburst=self.N1, burst=m * self.N0 // 16,
                                            src_stride=(self.Nq - m) * self.N0 // 16, dst_stride=0)

                broadcast_scale1 = self.broadcast(scale1, (m, n0))
                broadcast_scale1 = broadcast_scale1.reshape((1, m, n0))
                with self.tik_instance.for_range(0, n1) as idx:
                    self.tik_instance.h_mul(Oi_ub[idx, :, :], Oi_ub[idx, :, :], broadcast_scale1)

                # vec: scale2 * Pij_Vj
                broadcast_scale2 = self.broadcast(scale2, (m, n0))
                broadcast_scale2 = broadcast_scale2.reshape((1, m, n0))

                with self.tik_instance.for_range(0, n1) as idx:
                    self.tik_instance.h_mul(Pij_Vj_ub[idx, :, :], Pij_Vj_ub[idx, :, :], broadcast_scale2)

                self.tik_instance.h_add(Oi_ub, Oi_ub, Pij_Vj_ub)
                self.tik_instance.data_move(dst=self.O_gm[o_gm_offset], src=Oi_ub, sid=0,
                                            nburst=self.N1, burst=block_h * self.N0 // 16,
                                            src_stride=0, dst_stride=(self.Nq - h_alig) * self.N0 // 16)
            else:
                Oi_ub = self.tik_instance.Tensor(FP16, (h_alig, self.d), scope=UB, name="Oi_ub")
                self.tik_instance.data_move(
                    Oi_ub, self.O_gm[o_gm_offset], 0, 1, block_h * self.d // 16, 0, 0
                )
                Oi_new_ub = self.clac_new_Oi(Oi_ub, exp_mi_sub_mi_new, Pij_Vj_ub, exp_mij_sub_mi_new,
                                             li_new_rec_ub, li_ub, block_h, self.d)

                self.tik_instance.data_move(self.O_gm[o_gm_offset], Oi_new_ub, 0, 1, block_h * self.d // 16, 0, 0)

    def row_sum_cube_impl(self, Sij_l1_K1MK0_ed, lij_ub, m, k):
        """
        :param Sij_ub: the tensor with shape of (K1, M, K0) in ub
        :return:
        """
        K1, M, K0 = Sij_l1_K1MK0_ed.shape
        K = K1 * K0

        if self.front_construct_one_matrix:
            right_all_one_matrix_l1 = self.right_all_one_matrix_l1
        else:
            right_all_one_matrix_ub = self.tik_instance.Tensor(
                FP16, (K, 16), name="right_all_one_matrix_ub", scope=UB
            )
            self.tik_instance.h_duplicate(right_all_one_matrix_ub, 1.0)
            right_all_one_matrix_l1 = self.tik_instance.Tensor(
                FP16, (K1 * K0, 16), name="right_all_one_matrix_l1", scope=L1
            )
            self.tik_instance.data_move(right_all_one_matrix_l1,
                                        right_all_one_matrix_ub,
                                        0, 1, K, 0, 0)

        if self.auto_road_conversion:
            row_sum_ub_N1MN0 = self.matmul_compute_with_road_conversion(Sij_l1_K1MK0_ed, right_all_one_matrix_l1,
                                                                        m, k, 16, reorder_res=False)
        else:
            row_sum_ub = self.tik_instance.Tensor(FP16, (1, M, 16), name="row_sum_ub", scope=UB)
            row_sum_ub_N1MN0 = self.matmul_compute(Sij_l1_K1MK0_ed,
                                                   right_all_one_matrix_l1,
                                                   row_sum_ub,
                                                   m, k, 16, reorder_res=False)
        row_sum_ub_MN_ed = row_sum_ub_N1MN0.reshape((M, 16))
        row_sum_ub_trans = self.tik_instance.Tensor(FP16, (16, M), name="row_sum_ub_trans", scope=UB)
        row_sum_ub_trans = self.transpose_matrix(row_sum_ub_MN_ed, row_sum_ub_trans, M, True)
        self.tik_instance.data_move(lij_ub, row_sum_ub_trans[0, 0:m], 0, 1, m // 16, 0, 0)

        return lij_ub

    def softmax_without_row_max(self, Sij_ub, lij_ub, m, n):
        """
        :param Sij_ub:
        :param lij_ub:
        :return:
        """
        K1, M, K0 = Sij_ub.shape
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # P_ij = exp(Sij)
            self.tik_instance.h_exp(Sij_ub, Sij_ub)

            Sij_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (K1, M, K0), name="Sij_l1_K1MK0_ed",
                                                       scope=L1)
            self.tik_instance.data_move(Sij_l1_K1MK0_ed, Sij_ub, 0, 1, K1 * M * K0 // 16, 0, 0)
            lij_ub = self.row_sum_cube_impl(Sij_l1_K1MK0_ed, lij_ub, m, n)

        return Sij_ub, lij_ub

    def update_l_o_without_rowmax(self,
                                  Pij_Vj_ub,
                                  lij_ub,
                                  batch_start,
                                  batch_idx,
                                  kv_blk_idx,
                                  q_blk_idx,
                                  block_h
                                  ):
        vec_gm_offset = (batch_start + batch_idx) * self.Nq + q_blk_idx * self.Br
        o_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx)
        h_alig = self.up_align_to_K0(block_h)
        # li_new=lij. Oi <- Pij*Vj/Li_new
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
                        Pij_Vj_ub[half_block_offset + i, :],
                        src_scalar,
                        Pij_Vj_ub[half_block_offset + i, :],
                    )
                o_half_block_offset = o_gm_offset + half_block_offset * self.d
                tensor = Pij_Vj_ub[half_block_offset, :]
                self.tik_instance.data_move(self.O_gm[o_half_block_offset], tensor, 0, 1,
                                            half_block_h * self.d // 16, 0, 0)

        with self.tik_instance.else_scope():
            li_ub = self.tik_instance.Tensor(FP16, (block_h,), name="li_ub", scope=UB)
            self.move_vector_from_gm_to_ub(li_ub, self.l_gm, vec_gm_offset)
            self.tik_instance.h_add(lij_ub, li_ub, lij_ub)
            self.move_vector_from_ub_to_gm(self.l_gm, lij_ub, vec_gm_offset, block_h)

            with self.tik_instance.new_stmt_scope(disable_sync=False):
                # load Oi
                Oi_ub = self.tik_instance.Tensor(FP16, (h_alig, self.d), scope=UB, name="Oi_ub")
                self.tik_instance.data_move(
                    Oi_ub, self.O_gm[o_gm_offset], 0, 1, block_h * self.d // 16, 0, 0
                )
                with self.tik_instance.for_range(0, block_h) as i:
                    src_scalar = self.tik_instance.Scalar(init_value=li_ub[i], dtype=FP16)
                    self.tik_instance.h_mul(Oi_ub[i, :], Oi_ub[i, :], src_scalar)
                self.tik_instance.h_add(Pij_Vj_ub, Oi_ub, Pij_Vj_ub)
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
                        Pij_Vj_ub[half_block_offset + i, :],
                        src_scalar,
                        Pij_Vj_ub[half_block_offset + i, :],
                    )
                o_half_block_offset = o_gm_offset + half_block_offset * self.d
                src_tensor = Pij_Vj_ub[half_block_offset, :]
                self.tik_instance.data_move(self.O_gm[o_half_block_offset], src_tensor, 0, 1,
                                            half_block_h * self.d // 16, 0, 0)

    def up_align_to_K0(self, n, dtype=None):
        if dtype is None:
            dtype = self.dtype

        K0 = 32 // DTYPE_SIZE[dtype]
        return (n + K0 - 1) // K0 * K0

    def compute_in_each_kv_block(self, batch_start, batch_idx, kv_blk_idx, kv_blk_height,
                                 core_idx_to_tr_info=None, core_idx=None):
        Bc_alig = self.up_align_to_K0(kv_blk_height)
        k_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.N, self.d,
                                         self.Bc, kv_blk_idx)
        if self.input_with_n1mn0:
            Kj_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (Bc_alig, self.d), name="Kj_l1", scope=L1)
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                self.cont_data_mv_1_bust(dst=Kj_l1_K1MK0_ed, src=self.k_gm[k_gm_offset],
                                         burst=kv_blk_height * self.d // 16)
        elif self.data_with_nz:
            Kj_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (self.d // self.N0, Bc_alig, self.N0),
                                                      name="KjT_l1_K1MK0_ed", scope=L1)
            self.tik_instance.data_move(dst=Kj_l1_K1MK0_ed, src=self.k_gm[k_gm_offset],
                                        sid=0, nburst=self.N1, burst=Bc_alig * self.N0 // 16,
                                        src_stride=(self.N - Bc_alig) * self.N0 // 16, dst_stride=0)
        else:
            Kj_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (Bc_alig, self.d), name="Kj_l1", scope=L1)
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                Kj_ub = self.tik_instance.Tensor(FP16, (Bc_alig, self.d), name="Kj_ub", scope=UB)
                self.tik_instance.data_move(
                    Kj_ub, self.k_gm[k_gm_offset], 0, 1, kv_blk_height * self.d // 16, 0, 0
                )
                Kj_l1_K1MK0_ed = self.MK_TO_K1MK0(Kj_ub, workspace_tensor=Kj_l1_K1MK0_ed)
        # KN_TO_K1NK0: gm -> ub -> for (src_ub -> k1nk0_ub -> l1)
        # KN_TO_K1NK0_V2: gm -> ub -> ub -> scatter_ub -> l1
        Vj_l1_K1NK0_ed = self.tik_instance.Tensor(FP16, (Bc_alig, self.d), name="Vj_l1", scope=L1)
        v_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.N, self.d,
                                         self.Bc, kv_blk_idx)
        if self.input_with_n1mn0:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                self.cont_data_mv_1_bust(dst=Vj_l1_K1NK0_ed, src=self.v_gm[v_gm_offset],
                                         burst=kv_blk_height * self.d // 16)
        elif self.data_with_nz:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                Vj_ub = self.tik_instance.Tensor(FP16, (self.d // self.N0, Bc_alig, self.N0),
                                                 name="Vj_ub", scope=UB)
                self.tik_instance.data_move(dst=Vj_ub, src=self.v_gm[v_gm_offset],
                                            sid=0, nburst=self.N1, burst=Bc_alig * self.N0 // 16,
                                            src_stride=(self.N - Bc_alig) * self.N0 // 16, dst_stride=0)
                # (N1, K, N0) -> (K, N)
                Vj_ub = self.N1MN0_TO_MN(Vj_ub)
                # (K, N) -> (K1, N, K0)
                Vj_l1_K1NK0_ed = self.KN_TO_K1NK0(Vj_ub, workspace_tensor=Vj_l1_K1NK0_ed)
        else:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                Vj_ub = self.tik_instance.Tensor(FP16, (Bc_alig, self.d), name="Vj_ub", scope=UB)
                self.tik_instance.data_move(
                    Vj_ub, self.v_gm[v_gm_offset], 0, 1, kv_blk_height * self.d // 16, 0, 0
                )
                Vj_l1_K1NK0_ed = self.KN_TO_K1NK0(Vj_ub, workspace_tensor=Vj_l1_K1NK0_ed)

        if self.update_sub_core_strategy:
            tr_start_s = self.tik_instance.Scalar("int32", name="tr_start_s")
            tr_end_s = self.tik_instance.Scalar("int32", name="tr_end_s")
            tr_start_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 0])
            tr_end_s.set_as(core_idx_to_tr_info[core_idx, batch_start + batch_idx, 1])
            with self.tik_instance.for_range(tr_start_s, tr_end_s, name="q_blk_idx") as q_blk_idx:
                with self.tik_instance.if_scope(q_blk_idx != self.Tr - 1):
                    self.compute_in_each_q_block(Kj_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx,
                                                 batch_start,
                                                 kv_blk_height, self.Br, q_blk_idx, kv_blk_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_q_block(Kj_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx,
                                                 batch_start,
                                                 kv_blk_height, self.last_Br, q_blk_idx, kv_blk_idx)
        else:
            # for double buffer.
            with self.tik_instance.for_range(0, self.Tr // 2, name="q_blk_idx") as q_loop_idx:
                with self.tik_instance.for_range(begint=0, endt=2, thread_num=2) as t_id:
                    q_blk_idx = 2 * q_loop_idx + t_id
                    with self.tik_instance.if_scope(q_blk_idx != self.Tr - 1):
                        self.compute_in_each_q_block(Kj_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx,
                                                     batch_start,
                                                     kv_blk_height, self.Br, q_blk_idx, kv_blk_idx)
                    with self.tik_instance.else_scope():
                        self.compute_in_each_q_block(Kj_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx,
                                                     batch_start,
                                                     kv_blk_height, self.last_Br, q_blk_idx, kv_blk_idx)

    def compute_in_each_q_block(self, Kj_l1_K1MK0_ed, Vj_l1_K1NK0_ed, batch_idx, batch_start,
                                kv_blk_height, q_blk_height, q_blk_idx, kv_blk_idx):
        kv_blk_h_aligned = self.up_align_to_K0(kv_blk_height)
        q_blk_h_aligned = self.up_align_to_K0(q_blk_height)

        q_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx)
        if self.input_with_n1mn0:
            Qi_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (q_blk_h_aligned, self.d), scope=L1, name="Qi_l1")
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                self.cont_data_mv_1_bust(dst=Qi_l1_K1MK0_ed, src=self.q_gm[q_gm_offset],
                                         burst=q_blk_height * self.d // 16)
        elif self.data_with_nz:
            Qi_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (self.d // self.N0, q_blk_h_aligned, self.N0),
                                                      scope=L1, name="Qi_l1_K1MK0_ed")
            self.tik_instance.data_move(dst=Qi_l1_K1MK0_ed, src=self.q_gm[q_gm_offset],
                                        sid=0, nburst=self.N1, burst=q_blk_h_aligned * self.N0 // 16,
                                        src_stride=(self.Nq - q_blk_h_aligned) * self.N0 // 16, dst_stride=0)
        else:
            Qi_l1_K1MK0_ed = self.tik_instance.Tensor(FP16, (q_blk_h_aligned, self.d), scope=L1, name="Qi_l1")
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                Qi_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned, self.d), scope=UB, name="Qi_ub")
                self.tik_instance.data_move(
                    Qi_ub, self.q_gm[q_gm_offset], 0, 1, q_blk_height * self.d // 16, 0, 0
                )
                Qi_l1_K1MK0_ed = self.MK_TO_K1MK0(Qi_ub, workspace_tensor=Qi_l1_K1MK0_ed)

        if SOFTMAX_WITH_ROWMAX:
            mij_ub = self.tik_instance.Tensor(FP16, (q_blk_height,), scope=UB, name="mij_ub")
        lij_ub = self.tik_instance.Tensor(FP16, (q_blk_height,), scope=UB, name="lij_ub")
        # shape: (N1, M, N0) (N//16, M, 16)
        Pij_l1_K1MK0_ed = self.tik_instance.Tensor(
            FP16, (kv_blk_h_aligned // 16, q_blk_h_aligned, 16), name="Pij_l1", scope=L1
        )
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            reorder_res = True
            if not SOFTMAX_WITH_ROWMAX or self.softmax_with_nz or self.data_with_nz:
                reorder_res = False
            if self.auto_road_conversion:
                Sij_ub_MN_ed = self.matmul_compute_with_road_conversion(
                    Qi_l1_K1MK0_ed,
                    Kj_l1_K1MK0_ed,
                    m=q_blk_height,
                    k=self.actual_d,
                    n=kv_blk_height,
                    reorder_res=reorder_res
                )  # q*kT
            else:
                # N1MN0
                Sij_ub = self.tik_instance.Tensor(FP16, (kv_blk_h_aligned // 16, q_blk_h_aligned, 16), name="Sij_ub",
                                                  scope=UB)
                # QK^T Q shape: (q_blk_h_aligned, self.d), K^T shape: (self.d, kv_blk_h_aligned)
                # M, N (q_blk_h_aligned, kv_blk_h_aligned)
                Sij_ub_MN_ed = self.matmul_compute(
                    Qi_l1_K1MK0_ed,
                    Kj_l1_K1MK0_ed,
                    Sij_ub,
                    m=q_blk_height,
                    k=self.actual_d,
                    n=kv_blk_height,
                    reorder_res=reorder_res
                )  # q*kT

            Sij_ub_MN_ed_scaled = self.scale_compute_vector(Sij_ub_MN_ed, self.actual_d)

            if self.softmax_with_nz or self.data_with_nz:
                Pij_l1_K1MK0_ed, mij_ub, lij_ub = self.softmax_compute_with_nz(
                    Pij_l1_K1MK0_ed, Sij_ub_MN_ed_scaled, mij_ub, lij_ub, q_blk_height, kv_blk_height)
            elif self.remove_repeat_fractal:
                Pij_l1_K1MK0_ed, mij_ub, lij_ub = self.softmax_compute_with_fractal(
                    Pij_l1_K1MK0_ed, Sij_ub_MN_ed_scaled, mij_ub, lij_ub, q_blk_height, kv_blk_height)
            else:
                if SOFTMAX_WITH_ROWMAX:
                    Pij_ub, mij_ub, lij_ub = self.softmax_compute(
                        Sij_ub_MN_ed_scaled, mij_ub, lij_ub, q_blk_height, kv_blk_height)
                else:
                    Pij_ub, lij_ub = self.softmax_without_row_max(
                        Sij_ub_MN_ed_scaled, lij_ub, q_blk_height, kv_blk_height)
                # ub -> l1
                if reorder_res:
                    Pij_l1_K1MK0_ed = self.MK_TO_K1MK0(Pij_ub, workspace_tensor=Pij_l1_K1MK0_ed)
                else:
                    self.tik_instance.data_move(Pij_l1_K1MK0_ed,
                                                Pij_ub,
                                                0,
                                                1,
                                                q_blk_h_aligned * kv_blk_h_aligned // 16,
                                                0,
                                                0)
        if self.auto_road_conversion:
            if self.output_with_n1mn0 or self.data_with_nz:
                reorder_res = False
            Pij_Vj_matmul_res_ub = self.matmul_compute_with_road_conversion(
                Pij_l1_K1MK0_ed, Vj_l1_K1NK0_ed,
                q_blk_height, kv_blk_height, self.actual_d,
                reorder_res=reorder_res
            )
        else:
            # shape: (N1, M, N0) (N//16, M, 16)
            Pij_Vj_ub = self.tik_instance.Tensor(FP16, (self.d // 16, q_blk_h_aligned, 16), name="Pij_Vj_ub", scope=UB)
            # Pij_Vj_ub shape:             (M, K) (q_blk_h_aligned, kv_blk_h_aligned)
            # Vj_l1 shape:                 (K,N)  (kv_blk_h_aligned, self.d)
            # Pij_Vj_matmul_res_ub shape:  (M, N) (q_blk_h_aligned, self.d)
            Pij_Vj_matmul_res_ub = self.matmul_compute(
                Pij_l1_K1MK0_ed, Vj_l1_K1NK0_ed, Pij_Vj_ub, q_blk_height, kv_blk_height, self.actual_d,
                reorder_res=True
            )
        if SOFTMAX_WITH_ROWMAX:
            if self.output_with_n1mn0:
                self.update_o_m_l_n1mn0(
                    Pij_Vj_matmul_res_ub,
                    mij_ub,
                    lij_ub,
                    batch_start,
                    batch_idx,
                    kv_blk_idx,
                    q_blk_idx,
                    q_blk_height
                )
            else:
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
        else:
            self.update_l_o_without_rowmax(
                Pij_Vj_matmul_res_ub,
                lij_ub,
                batch_start,
                batch_idx,
                kv_blk_idx,
                q_blk_idx,
                q_blk_height
            )

    def compute_one_core(self, batch_start_sc, batch_num_sc, core_idx_to_tr_info=None, core_idx=None):
        with self.tik_instance.for_range(0, batch_num_sc, name="batch_index") as batch_idx:
            with self.tik_instance.for_range(0, self.Tc, name="kv_blk_idx") as kv_blk_idx:
                with self.tik_instance.if_scope(kv_blk_idx != self.Tc - 1):
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.Bc,
                                                  core_idx_to_tr_info, core_idx)
                with self.tik_instance.else_scope():
                    self.compute_in_each_kv_block(batch_start_sc, batch_idx, kv_blk_idx, self.last_Bc,
                                                  core_idx_to_tr_info, core_idx)

    def compute_process(self):
        self.init()
        if self.update_sub_core_strategy:
            if self.core_num > self.B * self.Tr:
                self.core_num = self.B * self.Tr
            core_idx_to_batch_info, core_idx_to_tr_info = self.get_each_core_task_info()
            with self.tik_instance.for_range(begint=0, endt=self.core_num, name="core_index",
                                             block_num=self.core_num) as core_idx:
                batch_start_s = self.tik_instance.Scalar("int32", name="batch_start_s")
                batch_num_s = self.tik_instance.Scalar("int32", name="batch_num_s")
                batch_start_s.set_as(core_idx_to_batch_info[core_idx, 0])
                batch_num_s.set_as(core_idx_to_batch_info[core_idx, 1])
                self.compute_one_core(batch_start_s, batch_num_s, core_idx_to_tr_info, core_idx)
        else:
            if self.core_num > self.B:
                self.core_num = self.B
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


def flash_attention(q, k, v, attention_mask, y, kernel_name="flash_attention", disable_debug=True):
    fa = FlashAttention(q=q, k=k, v=v, disable_debug=disable_debug)
    fa.compute_process()
    fa.tik_instance.BuildCCE(
        kernel_name=kernel_name,
        inputs=[fa.q_gm, fa.k_gm, fa.v_gm],
        outputs=[fa.O_gm],
        config={"dump_cce_code": False, "save_temp_cce_file": True, "enable_const_fold": True},
        enable_l2=True
    )
    return fa.tik_instance
