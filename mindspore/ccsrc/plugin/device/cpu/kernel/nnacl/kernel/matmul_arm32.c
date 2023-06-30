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

#ifdef ENABLE_ARM32
#include "nnacl/kernel/matmul_arm32.h"
#include "nnacl/kernel/matmul_base.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"

void MatmulARM32InitGlobalVariable(MatmulStruct *matmul) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  matmul->matrix_a_.need_pack_ = true;
  matmul->matrix_b_.need_pack_ = true;
  matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2Row12MajorParallel : RowMajor2Col12MajorParallel;
  matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2Col4MajorParallel : RowMajor2Row4MajorParallel;
  matmul->compute_.row_tile_ = C12NUM;
  matmul->compute_.col_tile_ = C4NUM;
  matmul->compute_.col_min_unit_ = C4NUM;
}

int MatmulARM32ParallelRunByBatch(MatmulStruct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MatmulComputeParam *compute = &matmul->compute_;
  ActType act = param->act_type_;

  int start_batch = task_id * compute->batch_stride_;
  int end_batch = MSMIN(matmul->batch_, start_batch + compute->batch_stride_);
  int func_flag = 0;
  if (compute->row_ == 1) {
    func_flag += (!matmul->b_const_ && compute->col_ <= C128NUM) ? C2NUM : C1NUM;
  }

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * compute->row_align_ * compute->deep_;
    const float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * compute->deep_ * compute->col_align_;
    float *c = matmul->output_data_ + index * compute->row_ * compute->col_step_;

    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_;
    if (func_flag == 0) {
      MatMulOpt(a, b, c, bias, act, compute->deep_, compute->row_, compute->col_step_, compute->col_, OutType_Nhwc);
    } else if (func_flag == C1NUM) {
      MatVecMulFp32Block4(a, b, c, bias, act, compute->deep_, compute->col_step_);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, act, compute->deep_, compute->col_step_, compute->col_step_);
    }
  }
  return NNACL_OK;
}

int MatmulARM32ParallelRunByOC(MatmulStruct *matmul, int task_id) {
  NNACL_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MatmulComputeParam *compute = &matmul->compute_;
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
      MatMulOpt(a, b, c, bias, act, compute->deep_, compute->row_, compute_oc, compute->col_, OutType_Nhwc);
    } else if (func_flag == C1NUM) {
      MatVecMulFp32Block4(a, b, c, bias, act, compute->deep_, compute_oc);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, act, compute->deep_, compute_oc, compute->col_step_);
    }
  }
  return NNACL_OK;
}

KernelBase *CreateMatmulARM32() {
  MatmulStruct *matmul = (MatmulStruct *)CreateMatmulBase();
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(matmul);
  matmul->matmul_type_ = kNotImplemented;
  matmul->init_global_varibale_ = MatmulARM32InitGlobalVariable;
  matmul->parallel_run_by_batch_ = MatmulARM32ParallelRunByBatch;
  matmul->parallel_run_by_oc_ = MatmulARM32ParallelRunByOC;
  return (KernelBase *)matmul;
}
#endif
