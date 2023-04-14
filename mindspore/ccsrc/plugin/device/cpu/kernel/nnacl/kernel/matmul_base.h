/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NNACL_KERNEL_MATMUL_BASE_H_
#define NNACL_KERNEL_MATMUL_BASE_H_

#include "nnacl/kernel.h"
#include "nnacl/matmul_parameter.h"

#define SPLIT_COUNT 32

typedef struct MatrixInfo {
  bool need_pack_;
  bool has_packed_;  // only valid for constant, only do once throughout the process.
  bool has_origin_;  // only valid for constant, only true when failing to infer shape, then false after packed.
  int pack_size_;
  float *origin_ptr_;  // only valid for constant, which is synchronized with the 'has_origin'.
  float *pack_ptr_;
} MatrixInfo;

typedef struct MatmulSlice {
  int row_s_;
  int row_e_;
  int col_s_;
  int col_e_;
} MatmulSlice;

typedef struct MatmulFp32Struct {
  KernelBase base_;
  int row_;
  int col_;
  int deep_;
  int row_align_;
  int col_align_;
  int deep_align_;
  int row_num_;
  int col_tile_;
  int row_tile_;
  int col_step_;
  int row_min_unit_;
  int col_min_unit_;
  int batch_stride_;
  int pack_b_stride_;
  int block_col_unit_;
  MatmulType matmul_type_;

  /* model pool optimize */
  int model_thread_nr_;

  /* batch-matmul broadcast */
  int batch_;
  int a_batch_;
  int b_batch_;
  int *a_offset_; /* batch_ size */
  int *b_offset_; /* batch_ size */

  int split_points_[SPLIT_COUNT];

  float *output_data_;
  float *conv1x1_origin_bias_;
  float *conv1x1_origin_weight_;
  float *pack_b_src_;
  float *pack_b_dst_;

  bool a_const_;
  bool b_const_;
  bool infer_shape_;
  bool pack_opt_;
  bool is_sharing_pack_;
  bool out_need_aligned_;
  bool weight_is_packed_;
  bool support_mul_batch_cut_by_row_;

  MatrixInfo matrix_a_;
  MatrixInfo matrix_b_;
  MatrixInfo matrix_c_;

  void (*matrix_a_pack_fun_)(const float *src_ptr, float *dst_ptr, int row, int col, int start_row, int end_row);
  void (*matrix_b_pack_fun_)(const float *src_ptr, float *dst_ptr, int row, int col, int start_row, int end_row);

  int (*pack_matrix_a_impl_opt_)(struct MatmulFp32Struct *matmul);
  int (*pack_matrix_a_impl_)(struct MatmulFp32Struct *matmul);
  int (*pack_matrix_b_impl_)(struct MatmulFp32Struct *matmul);

  int (*init_parameter_)(struct MatmulFp32Struct *matmul);
  void (*init_global_varibale_)(struct MatmulFp32Struct *matmul);

  bool (*check_thread_cutting_by_row_)(struct MatmulFp32Struct *matmul);
  void (*get_thread_cutting_policy_)(struct MatmulFp32Struct *matmul);
  void (*get_thread_cutting_info_by_row_)(struct MatmulFp32Struct *matmul);

  void *pack_weight_manager_;
  void *(*get_pack_data_by_sharing_weight_)(void *manager, const void *tensor_data, const size_t size, bool *is_packed);
  void (*free_by_sharing_weight_)(void *manager, void *tensor_data);

  void (*gemm_not_pack_fun_)(const float *a, const float *b, float *c, const float *bias, int m, int k, int act_type);

  int (*parallel_run_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_by_row_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_by_oc_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_by_batch_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_not_pack_by_batch_)(struct MatmulFp32Struct *matmul, int task_id);

  /* optimize for avx512 */
  int col_split_points_size_;
  int row_split_points_size_;
  int col_split_points_[SPLIT_COUNT];
  int row_split_points_[SPLIT_COUNT];
  int matmul_slice_count_[SPLIT_COUNT];
  MatmulSlice matmul_slice_set_[SPLIT_COUNT][SPLIT_COUNT];
  int (*parallel_run_by_gemm_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_by_gepm_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_by_gepdot_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_by_batch_col_row_gemm_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_by_row1_deep1_gepdot_)(struct MatmulFp32Struct *matmul, int task_id);
} MatmulFp32Struct;

#endif  // NNACL_KERNEL_MATMUL_BASE_H_
