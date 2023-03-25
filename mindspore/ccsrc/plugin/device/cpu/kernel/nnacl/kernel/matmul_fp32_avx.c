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

#ifdef ENABLE_AVX
#include "nnacl/kernel/matmul_fp32_avx.h"
#include "nnacl/kernel/matmul_fp32_base.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

void MatmulFp32Avx_InitGlobalVariable(MatmulFp32Struct *matmul) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  matmul->row_tile_ = C1NUM;
  matmul->col_tile_ = C8NUM;
  matmul->col_min_unit_ = C32NUM;
  matmul->out_need_aligned_ = true;
  matmul->matrix_b_.need_pack_ = true;
  matmul->matrix_a_.need_pack_ = param->a_transpose_;
  matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
  matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2Col32MajorParallel : RowMajor2Row32MajorParallel;
}

int MatmulFp32Avx_PackMatrixAImplOpt(MatmulFp32Struct *matmul) { return NNACL_ERR; }

int MatmulFp32Avx_ParallelRunByBatch(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  int start_batch = task_id * matmul->batch_stride_;
  int end_batch = MSMIN(matmul->batch_, start_batch + matmul->batch_stride_);
  int func_flag = 0;
  if (matmul->row_ == 1) {
    func_flag += (!matmul->b_const_ && matmul->col_ <= C128NUM) ? C2NUM : C1NUM;
  }

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * matmul->row_align_ * matmul->deep_;
    const float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * matmul->deep_ * matmul->col_align_;
    float *c = matmul->output_data_ + index * matmul->row_ * matmul->col_step_;

    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_;
    if (func_flag == 0) {
      MatMulAvxFp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_, matmul->col_align_,
                    matmul->row_);
    } else if (func_flag == C1NUM) {
      MatVecMulAvxFp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_, matmul->col_align_);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_, matmul->col_step_);
    }
  }
  return NNACL_OK;
}

int MatmulFp32Avx_ParallelRunByRow(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MS_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);

  int start_row = matmul->split_points_[task_id];
  int end_row = matmul->row_num_;
  if (task_id < (matmul->base_.thread_nr_ - 1)) {
    end_row = matmul->split_points_[task_id + 1];
  }
  int row_num = end_row - start_row;
  if (row_num <= 0) {
    return NNACL_OK;
  }
  const float *input = matmul->matrix_a_.pack_ptr_ + start_row * matmul->deep_;
  float *output = matmul->output_data_ + start_row * matmul->col_align_;
  if (matmul->col_ == 1) {
    float bias = 0;
    if (matmul->matrix_c_.pack_ptr_ != NULL) {
      bias = matmul->matrix_c_.pack_ptr_[0];
    }
    matmul->gemm_not_pack_fun_(input, matmul->matrix_b_.pack_ptr_, output, &bias, row_num, matmul->deep_,
                               param->act_type_);
  } else {
    MatMulAvxFp32(input, matmul->matrix_b_.pack_ptr_, output, matmul->matrix_c_.pack_ptr_, param->act_type_,
                  matmul->deep_, matmul->col_align_, matmul->col_align_, row_num);
  }
  return NNACL_OK;
}

int MatmulFp32Avx_ParallelRunByOC(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MS_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);

  int start_oc = matmul->split_points_[task_id];
  int end_oc = matmul->col_step_;
  if (task_id < (matmul->base_.thread_nr_ - 1)) {
    end_oc = matmul->split_points_[task_id + 1];
  }
  int compute_oc = end_oc - start_oc;
  if (compute_oc <= 0) {
    return NNACL_OK;
  }
  int func_flag = 0;
  if (matmul->row_ == 1) {
    func_flag += (!matmul->b_const_ && matmul->col_ <= C128NUM) ? C2NUM : C1NUM;
  }
  int b_stride = func_flag == C2NUM ? 1 : matmul->deep_;
  for (int i = 0; i < matmul->batch_; ++i) {
    float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[i] * matmul->row_align_ * matmul->deep_;
    float *b =
      matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[i] * matmul->deep_ * matmul->col_align_ + start_oc * b_stride;
    float *c = matmul->output_data_ + i * matmul->row_ * matmul->col_step_ + start_oc;
    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_ + start_oc;
    if (func_flag == 0) {
      MatMulAvxFp32(a, b, c, bias, param->act_type_, matmul->deep_, compute_oc, matmul->col_align_, matmul->row_);
    } else if (func_flag == C1NUM) {
      MatVecMulAvxFp32(a, b, c, bias, param->act_type_, matmul->deep_, compute_oc, matmul->col_align_);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, matmul->deep_, compute_oc, matmul->col_step_);
    }
  }
  return NNACL_OK;
}

bool MatmulFp32Avx_CheckThreadCuttingByRow(MatmulFp32Struct *matmul) {
  if (matmul->b_batch_ != C1NUM) {
    return false;
  }
  if (matmul->row_num_ < matmul->base_.thread_nr_) {
    return false;
  }
  if (matmul->col_ == 1) {
    matmul->row_min_unit_ = C4NUM;
    return true;
  }
  if (matmul->row_ == 1 && !matmul->b_const_ && matmul->col_ <= C128NUM) {
    return false;
  }
  matmul->row_min_unit_ = C3NUM;
  if (matmul->col_step_ < C16NUM) {
    matmul->row_min_unit_ = C8NUM;
  } else if (matmul->col_step_ < C24NUM) {
    matmul->row_min_unit_ = C6NUM;
  } else if (matmul->col_step_ < C32NUM) {
    matmul->row_min_unit_ = C4NUM;
  }
  return MSMIN(matmul->row_num_ / matmul->row_min_unit_, matmul->base_.thread_nr_) >
         MSMIN(matmul->col_step_ / matmul->col_min_unit_, matmul->base_.thread_nr_);
}

KernelBase *CreateMatmulFp32Avx() {
  MatmulFp32Struct *matmul = (MatmulFp32Struct *)CreateMatmulFp32Base();
  matmul->init_global_varibale_ = MatmulFp32Avx_InitGlobalVariable;
  matmul->pack_matrix_a_impl_opt_ = MatmulFp32Avx_PackMatrixAImplOpt;
  matmul->parallel_run_by_batch_ = MatmulFp32Avx_ParallelRunByBatch;
  matmul->parallel_run_by_row_ = MatmulFp32Avx_ParallelRunByRow;
  matmul->parallel_run_by_oc_ = MatmulFp32Avx_ParallelRunByOC;
  matmul->check_thread_cutting_by_row_ = MatmulFp32Avx_CheckThreadCuttingByRow;
  return (KernelBase *)matmul;
}
#endif
