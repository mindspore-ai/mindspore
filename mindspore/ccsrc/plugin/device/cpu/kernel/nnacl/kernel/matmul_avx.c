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
#include "nnacl/kernel/matmul_avx.h"
#include "nnacl/kernel/matmul_base.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

void MatmulAVXInitGlobalVariable(MatmulStruct *matmul) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  matmul->compute_.row_tile_ = C1NUM;
  matmul->compute_.col_tile_ = C8NUM;
  matmul->compute_.col_min_unit_ = C32NUM;
  matmul->out_need_aligned_ = true;
  matmul->matrix_b_.need_pack_ = true;
  matmul->matrix_a_.need_pack_ = param->a_transpose_;
  matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
  matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2Col32MajorParallel : RowMajor2Row32MajorParallel;
}

int MatmulAVXParallelRunByBatch(MatmulStruct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)matmul->base_.param_;
  MatmulComputeParam *compute = (MatmulComputeParam *)&matmul->compute_;

  int start_batch = task_id * compute->batch_stride_;
  int end_batch = MSMIN(matmul->batch_, start_batch + compute->batch_stride_);
  int func_flag = 0;
  if (matmul->compute_.row_ == 1) {
    func_flag += (!matmul->b_const_ && compute->col_ <= C128NUM) ? C2NUM : C1NUM;
  }

  ActType act = param->act_type_;
  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * compute->row_align_ * compute->deep_;
    const float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * compute->deep_ * compute->col_align_;
    float *c = matmul->output_data_ + index * compute->row_ * compute->col_step_;

    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_;
    if (func_flag == 0) {
      MatMulAvxFp32(a, b, c, bias, act, compute->deep_, compute->col_step_, compute->col_align_, compute->row_);
    } else if (func_flag == C1NUM) {
      MatVecMulAvxFp32(a, b, c, bias, act, compute->deep_, compute->col_step_, compute->col_align_);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, act, compute->deep_, compute->col_step_, compute->col_step_);
    }
  }
  return NNACL_OK;
}

int MatmulAVXParallelRunByRow(MatmulStruct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  NNACL_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);
  MatmulComputeParam *compute = (MatmulComputeParam *)&matmul->compute_;

  int start_row = matmul->split_points_[task_id];
  int end_row = compute->row_num_;
  if (task_id < (matmul->base_.thread_nr_ - 1)) {
    end_row = matmul->split_points_[task_id + 1];
  }
  int row_num = end_row - start_row;
  if (row_num <= 0) {
    return NNACL_OK;
  }
  const float *input = matmul->matrix_a_.pack_ptr_ + start_row * compute->deep_;
  float *output = matmul->output_data_ + start_row * compute->col_align_;
  if (compute->col_ == 1) {
    float bias = 0;
    if (matmul->matrix_c_.pack_ptr_ != NULL) {
      bias = matmul->matrix_c_.pack_ptr_[0];
    }
    matmul->gemm_not_pack_fun_(input, matmul->matrix_b_.pack_ptr_, output, &bias, row_num, compute->deep_,
                               param->act_type_);
  } else {
    MatMulAvxFp32(input, matmul->matrix_b_.pack_ptr_, output, matmul->matrix_c_.pack_ptr_, param->act_type_,
                  compute->deep_, compute->col_align_, compute->col_align_, row_num);
  }
  return NNACL_OK;
}

int MatmulAVXParallelRunByOC(MatmulStruct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  NNACL_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);
  MatmulComputeParam *compute = (MatmulComputeParam *)&matmul->compute_;
  ActType act = param->act_type_;

  int start_oc = matmul->split_points_[task_id];
  int end_oc = compute->col_step_;
  if (task_id < (matmul->base_.thread_nr_ - 1)) {
    end_oc = matmul->split_points_[task_id + 1];
  }
  int compute_oc = end_oc - start_oc;
  if (compute_oc <= 0) {
    return NNACL_OK;
  }
  int func_flag = 0;
  if (compute->row_ == 1) {
    func_flag += (!matmul->b_const_ && compute->col_ <= C128NUM) ? C2NUM : C1NUM;
  }
  int b_stride = func_flag == C2NUM ? start_oc : start_oc * compute->deep_;

  for (int i = 0; i < matmul->batch_; ++i) {
    float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[i] * compute->row_align_ * compute->deep_;
    float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[i] * compute->deep_ * compute->col_align_ + b_stride;
    float *c = matmul->output_data_ + i * compute->row_ * compute->col_step_ + start_oc;
    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_ + start_oc;

    if (func_flag == 0) {
      MatMulAvxFp32(a, b, c, bias, param->act_type_, compute->deep_, compute_oc, compute->col_align_, compute->row_);
    } else if (func_flag == C1NUM) {
      MatVecMulAvxFp32(a, b, c, bias, act, compute->deep_, compute_oc, compute->col_align_);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, act, compute->deep_, compute_oc, compute->col_step_);
    }
  }
  return NNACL_OK;
}

bool MatmulAVXCheckThreadCuttingByRow(MatmulStruct *matmul) {
  if (matmul->b_batch_ != C1NUM) {
    return false;
  }
  if (matmul->compute_.row_num_ < matmul->base_.thread_nr_) {
    return false;
  }
  if (matmul->compute_.col_ == 1) {
    matmul->compute_.row_min_unit_ = C4NUM;
    return true;
  }
  if (matmul->compute_.row_ == 1 && !matmul->b_const_ && matmul->compute_.col_ <= C128NUM) {
    return false;
  }
  matmul->compute_.row_min_unit_ = C3NUM;
  if (matmul->compute_.col_step_ < C16NUM) {
    matmul->compute_.row_min_unit_ = C8NUM;
  } else if (matmul->compute_.col_step_ < C24NUM) {
    matmul->compute_.row_min_unit_ = C6NUM;
  } else if (matmul->compute_.col_step_ < C32NUM) {
    matmul->compute_.row_min_unit_ = C4NUM;
  }
  return MSMIN(matmul->compute_.row_num_ / matmul->compute_.row_min_unit_, matmul->base_.thread_nr_) >
         MSMIN(matmul->compute_.col_step_ / matmul->compute_.col_min_unit_, matmul->base_.thread_nr_);
}

KernelBase *CreateMatmulAVX() {
  MatmulStruct *matmul = (MatmulStruct *)CreateMatmulBase();
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(matmul);
  matmul->matmul_type_ = kNotImplemented;
  matmul->init_global_varibale_ = MatmulAVXInitGlobalVariable;
  matmul->parallel_run_by_batch_ = MatmulAVXParallelRunByBatch;
  matmul->parallel_run_by_row_ = MatmulAVXParallelRunByRow;
  matmul->parallel_run_by_oc_ = MatmulAVXParallelRunByOC;
  matmul->check_thread_cutting_by_row_ = MatmulAVXCheckThreadCuttingByRow;
  return (KernelBase *)matmul;
}
#endif
