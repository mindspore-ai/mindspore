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

#ifdef ENABLE_SSE
#include "nnacl/kernel/matmul_fp32_sse.h"
#include "nnacl/kernel/matmul_fp32_base.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

bool MatmulFp32Sse_CheckThreadCuttingByRow(MatmulFp32Struct *matmul) { return false; }
int MatmulFp32Sse_PackMatrixAImplOpt(MatmulFp32Struct *matmul) { return NNACL_ERR; }
int MatmulFp32Sse_ParallelRunByRow(MatmulFp32Struct *matmul, int task_id) { return NNACL_ERR; }

void MatmulFp32Sse_InitGlobalVariable(MatmulFp32Struct *matmul) {
  MatMulParameter *param = (MatMulParameter *)matmul->base_.param_;
  matmul->matrix_a_.need_pack_ = true;
  matmul->matrix_b_.need_pack_ = true;
  matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2Row4MajorParallel : RowMajor2Col4MajorParallel;
  matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2Col8MajorParallel : RowMajor2Row8MajorParallel;
  matmul->row_tile_ = C4NUM;
  matmul->col_tile_ = C8NUM;
  matmul->col_min_unit_ = C8NUM;
}

int MatmulFp32Sse_ParallelRunByBatch(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)matmul->base_.param_;

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
      MatMulOpt(a, b, c, bias, param->act_type_, matmul->deep_, matmul->row_, matmul->col_step_, matmul->col_,
                OutType_Nhwc);
    } else if (func_flag == C1NUM) {
      MatVecMulFp32Block8(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_, matmul->col_step_);
    }
  }
  return NNACL_OK;
}

int MatmulFp32Sse_ParallelRunByOC(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)matmul->base_.param_;
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
      MatMulOpt(a, b, c, bias, param->act_type_, matmul->deep_, matmul->row_, compute_oc, matmul->col_, OutType_Nhwc);
    } else if (func_flag == C1NUM) {
      MatVecMulFp32Block8(a, b, c, bias, param->act_type_, param->deep_, compute_oc);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, param->deep_, compute_oc, matmul->col_step_);
    }
  }
  return NNACL_OK;
}

KernelBase *CreateMatmulFp32Sse() {
  MatmulFp32Struct *matmul = (MatmulFp32Struct *)CreateMatmulFp32Base();
  matmul->check_thread_cutting_by_row_ = MatmulFp32Sse_CheckThreadCuttingByRow;
  matmul->pack_matrix_a_impl_opt_ = MatmulFp32Sse_PackMatrixAImplOpt;
  matmul->init_global_varibale_ = MatmulFp32Sse_InitGlobalVariable;
  matmul->parallel_run_by_oc_ = MatmulFp32Sse_ParallelRunByOC;
  matmul->parallel_run_by_row_ = MatmulFp32Sse_ParallelRunByRow;
  matmul->parallel_run_by_batch_ = MatmulFp32Sse_ParallelRunByBatch;
  return (KernelBase *)matmul;
}
#endif
