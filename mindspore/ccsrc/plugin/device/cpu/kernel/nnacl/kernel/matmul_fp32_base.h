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

#ifndef MINDSPORE_NNACL_KERNEL_MATMUL_FP32_BASE_H_
#define MINDSPORE_NNACL_KERNEL_MATMUL_FP32_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/matmul_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MatrixInfo {
  bool need_pack_;
  bool has_packed_;  // only valid for constant, only do once throughout the process.
  bool has_origin_;  // only valid for constant, only true when failing to infer shape, then false after packed.
  int pack_size_;
  float *origin_ptr_;  // only valid for constant, which is synchronized with the 'has_origin'.
  float *pack_ptr_;
} MatrixInfo;

typedef struct MatmulFp32Struct {
  KernelBase base_;
  int a_batch_;
  int b_batch_;
  int row_num_;
  int col_tile_;
  int row_tile_;
  int col_step_;
  int row_min_unit_;
  int col_min_unit_;
  int batch_stride_;
  int pack_b_stride_;

  int row_;
  int col_;
  int row_4_;
  int row_6_;
  int row_12_;
  int row_16_;
  int row_align_;
  int col_4_;
  int col_8_;
  int col_align_;
  int deep_;
  int deep_4_;
  int deep_16_;
  int deep_align_;
  int batch_;
  bool a_const_;
  bool b_const_;
  bool use_axis_;
  int axis_;

  bool infer_shape_;
  MatmulType matmul_type_;

  int model_thread_nr_;

  int split_points_[16];
  int a_offset_[512];
  int b_offset_[512];

  float *output_data_;
  float *conv1x1_origin_bias_;
  float *conv1x1_origin_weight_;
  float *pack_b_src_;
  float *pack_b_dst_;

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
  void (*init_global_varibale_)(struct MatmulFp32Struct *matmul);

  bool (*check_thread_cutting_by_row_)(struct MatmulFp32Struct *matmul);
  int (*get_thread_cutting_policy_)(struct MatmulFp32Struct *matmul);
  void (*get_thread_cutting_info_by_row_)(struct MatmulFp32Struct *matmul);

  void (*get_pack_data_by_sharing_weight_)(const void *tensor_data, const size_t size, bool *is_packed);
  void (*free_by_sharing_weight_)(void *tensor_data);

  int (*parallel_run_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_by_row_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_by_oc_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_by_batch_)(struct MatmulFp32Struct *matmul, int task_id);
  int (*parallel_run_not_pack_by_batch_)(struct MatmulFp32Struct *matmul, int task_id);

  void (*gemm_not_pack_fun_)(const float *a, const float *b, float *c, const float *bias, int m, int k, int act_type);
} MatmulFp32Struct;

int matmul_fp32_prepare(KernelBase *self);
int matmul_fp32_resize(KernelBase *self);
KernelBase *CreateMatmulFp32();
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_KERNEL_MATMUL_FP32_BASE_H_
