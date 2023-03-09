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

#ifdef ENABLE_ARM64
#include "nnacl/kernel/matmul_f32_arm64.h"
#include "nnacl/kernel/matmul_f32_base.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/pack_fp32_opt.h"

typedef struct MatrixAPack {
  int64_t points_[8];
  int64_t unit_num_;
  int thread_;
  int deep_;
  int row_;
  int col_;
  MatrixInfo *matrix_a_;
  float *src_ptr_;
  bool a_transpose_;
} MatrixAPack;

int MatmulFp32Arm64_PackMatrixAImplOptPack(void *cdata, int task_id, float l, float r) {
  MatrixAPack *pack = (MatrixAPack *)cdata;
  int64_t start = pack->points_[task_id];
  int64_t end = pack->unit_num_;
  if (task_id < pack->thread_ - 1) {
    end = pack->points_[task_id + 1];
  }

  if (pack->a_transpose_) {
    RowMajor2Row12MajorOpt(pack->src_ptr_, pack->matrix_a_->pack_ptr_, pack->deep_, pack->row_, start, end);
  } else {
    RowMajor2Col12MajorOpt(pack->src_ptr_, pack->matrix_a_->pack_ptr_, pack->row_, pack->deep_, start, end);
  }
  return NNACL_OK;
}

int MatmulFp32Arm64_PackMatrixAImplOpt(MatmulFp32Struct *matmul) {
  int64_t kPackAMinUnitNum = 1 << 13;
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  float *src_ptr =
    matmul->matrix_a_.has_origin_ ? matmul->matrix_a_.origin_ptr_ : (float *)(matmul->base_.in_[FIRST_INPUT]->data_);
  MS_CHECK_TRUE_RET(src_ptr != NULL, NNACL_ERR);
  MS_CHECK_TRUE_RET(matmul->matrix_a_.pack_ptr_ != NULL, NNACL_ERR);

  MatrixAPack pack;
  pack.src_ptr_ = src_ptr;
  pack.matrix_a_ = &matmul->matrix_a_;
  pack.deep_ = matmul->deep_;
  pack.col_ = matmul->col_;
  pack.row_ = matmul->row_;
  pack.a_transpose_ = param->a_transpose_;
  pack.unit_num_ = 0;
  pack.unit_num_ = matmul->a_batch_ * UP_DIV(matmul->row_, C12NUM) * matmul->deep_;
  pack.thread_ = MSMIN(matmul->base_.thread_nr_, UP_DIV(pack.unit_num_, kPackAMinUnitNum));
  if (pack.thread_ < 1) {
    pack.thread_ = 1;
  }
  int64_t block_size = pack.unit_num_ / pack.thread_;
  int64_t remain_size = pack.unit_num_ - block_size * pack.thread_;
  int64_t start = 0;
  size_t count = 0;
  while (start < pack.unit_num_) {
    pack.points_[count++] = start;
    start += block_size;
    if (remain_size > 0) {
      ++start;
      --remain_size;
    }
  }
  pack.thread_ = count;

  if (pack.thread_ == 1) {
    return MatmulFp32Arm64_PackMatrixAImplOptPack(&pack, 0, 0, 1);
  }
  return matmul->base_.env_->parallel_launch(matmul->base_.env_->thread_pool_, MatmulFp32Arm64_PackMatrixAImplOptPack,
                                             &pack, pack.thread_);
}

bool MatmulFp32Arm64_CheckThreadCuttingByRow(MatmulFp32Struct *matmul) {
  if (matmul->b_batch_ != C1NUM) {
    return false;
  }
  if (matmul->batch_ >= matmul->base_.thread_nr_ || matmul->col_ == 1) {
    matmul->row_min_unit_ = C4NUM;
    return true;
  }
  return false;
}
void MatmulFp32Arm64_InitGlobalVariable(MatmulFp32Struct *matmul) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  matmul->pack_opt_ = true;
  matmul->row_tile_ = C12NUM;
  matmul->col_tile_ = C8NUM;
  matmul->col_min_unit_ = C8NUM;
  matmul->matrix_a_.need_pack_ = true;
  matmul->matrix_b_.need_pack_ = !matmul->weight_is_packed_;
  matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2Row12MajorParallel : RowMajor2Col12MajorParallel;
  matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2Col8MajorParallel : RowMajor2Row8MajorParallel;
}

int MatmulFp32Arm64_ParallelRunByBatch(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MS_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);

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
      MatVecMulPackFp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_, matmul->col_step_);
    }
  }
  return NNACL_OK;
}

int MatmulFp32Arm64_ParallelRunByRow(MatmulFp32Struct *matmul, int task_id) {
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
  GemmIsNotPackByRow(matmul->matrix_a_.pack_ptr_, matmul->matrix_b_.pack_ptr_, matmul->output_data_,
                     matmul->matrix_c_.pack_ptr_, start_row, end_row, matmul->deep_, param->act_type_);
  return NNACL_OK;
}

int MatmulFp32Arm64_ParallelRunByOC(MatmulFp32Struct *matmul, int task_id) {
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
      MatMulOpt(a, b, c, bias, param->act_type_, matmul->deep_, matmul->row_, compute_oc, matmul->col_, OutType_Nhwc);
    } else if (func_flag == C1NUM) {
      MatVecMulPackFp32(a, b, c, bias, param->act_type_, matmul->deep_, compute_oc);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, matmul->deep_, compute_oc, matmul->col_step_);
    }
  }
  return NNACL_OK;
}

KernelBase *CreateMatmulFp32Arm64() {
  MatmulFp32Struct *matmul = (MatmulFp32Struct *)CreateMatmulFp32Base();
  matmul->matmul_type_ = kMatmulFp32Arm64Cpu;
  matmul->check_thread_cutting_by_row_ = MatmulFp32Arm64_CheckThreadCuttingByRow;
  matmul->init_global_varibale_ = MatmulFp32Arm64_InitGlobalVariable;
  matmul->parallel_run_by_oc_ = MatmulFp32Arm64_ParallelRunByOC;
  matmul->parallel_run_by_row_ = MatmulFp32Arm64_ParallelRunByRow;
  matmul->parallel_run_by_batch_ = MatmulFp32Arm64_ParallelRunByBatch;
  matmul->pack_matrix_a_impl_opt_ = MatmulFp32Arm64_PackMatrixAImplOpt;
  return (KernelBase *)matmul;
}
#endif
