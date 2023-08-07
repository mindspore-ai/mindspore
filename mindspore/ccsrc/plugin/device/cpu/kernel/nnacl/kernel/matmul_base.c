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

#include "nnacl/kernel/matmul_base.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/op_base.h"

#define kNumDeepThreshold 512

int MatmulFp32Run(void *cdata, int task_id, float l, float r) {
  NNACL_CHECK_NULL_RETURN_ERR(cdata);
  MatmulStruct *matmul = (MatmulStruct *)cdata;
  return matmul->parallel_run_(matmul, task_id);
}

void MatmulBaseFreeBatchOffset(MatmulStruct *matmul) {
  if (matmul->a_offset_ != NULL) {
    free(matmul->a_offset_);
    matmul->a_offset_ = NULL;
  }
  if (matmul->b_offset_ != NULL) {
    free(matmul->b_offset_);
    matmul->b_offset_ = NULL;
  }
}

int MatmulBaseMallocBatchOffset(MatmulStruct *matmul) {
  matmul->a_offset_ = malloc(matmul->batch_ * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(matmul->a_offset_);
  memset(matmul->a_offset_, 0, matmul->batch_ * sizeof(int));

  matmul->b_offset_ = malloc(matmul->batch_ * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(matmul->b_offset_);
  memset(matmul->b_offset_, 0, matmul->batch_ * sizeof(int));
  return NNACL_OK;
}

int MatmulBasePackMatrixBParallelRunByBatch(MatmulStruct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MatmulComputeParam *compute = &matmul->compute_;

  int start = task_id * compute->pack_b_stride_;
  if (param->b_transpose_) {
    int end = NNACL_MIN(matmul->compute_.col_, start + compute->pack_b_stride_);
    matmul->matrix_b_pack_fun_(matmul->pack_b_src_, matmul->pack_b_dst_, compute->col_, compute->deep_, start, end);
  } else {
    int end = NNACL_MIN(matmul->compute_.deep_, start + compute->pack_b_stride_);
    matmul->matrix_b_pack_fun_(matmul->pack_b_src_, matmul->pack_b_dst_, compute->deep_, compute->col_, start, end);
  }
  return NNACL_OK;
}

int MatmulFp32PackMatrixBRun(void *cdata, int task_id, float l, float r) {
  NNACL_CHECK_NULL_RETURN_ERR(cdata);
  MatmulStruct *matmul = (MatmulStruct *)cdata;
  return MatmulBasePackMatrixBParallelRunByBatch(matmul, task_id);
}

bool MatmulBaseCheckRowOptimalConditions(MatmulStruct *matmul) {
  return matmul->compute_.row_ == 1 &&
         !(matmul->support_mul_batch_cut_by_row_ && (matmul->a_batch_ > 1 && matmul->b_batch_ == 1));
}

int MatmulBaseInitParameter(MatmulStruct *matmul) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MatmulComputeParam *compute = &matmul->compute_;

  matmul->init_global_varibale_(matmul);
  if (MatmulBaseCheckRowOptimalConditions(matmul)) {
    compute->row_tile_ = 1;
    matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matmul->matrix_a_.need_pack_ = false;
    matmul->pack_opt_ = false;
    if (!matmul->b_const_ && compute->col_ <= C128NUM) {
      compute->col_tile_ = 1;
      matmul->out_need_aligned_ = false;
      matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
      matmul->matrix_b_.need_pack_ = param->b_transpose_;
    }
  }
  if (compute->col_ == 1 && !matmul->a_const_) {
    matmul->out_need_aligned_ = false;
    compute->row_tile_ = 1;
    compute->col_tile_ = 1;
    matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matmul->matrix_a_.need_pack_ = param->a_transpose_ && compute->row_ != 1;
    matmul->matrix_b_.need_pack_ = false;
    matmul->pack_opt_ = false;
  }
  compute->row_align_ = UP_ROUND(compute->row_, compute->row_tile_);
  compute->col_align_ = UP_ROUND(compute->col_, compute->col_tile_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(matmul->a_batch_, compute->row_align_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(matmul->a_batch_ * compute->row_align_, compute->deep_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(matmul->a_batch_, compute->col_align_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(matmul->a_batch_ * compute->col_align_, compute->deep_, NNACL_ERR);
  int a_pack_size = matmul->a_batch_ * compute->row_align_ * compute->deep_;
  int b_pack_size = matmul->b_batch_ * compute->col_align_ * compute->deep_;
  if ((matmul->matrix_a_.has_packed_ && matmul->matrix_a_.pack_size_ != a_pack_size) ||
      (matmul->matrix_b_.has_packed_ && matmul->matrix_b_.pack_size_ != b_pack_size)) {
    return NNACL_ERR;
  }
  matmul->matrix_a_.pack_size_ = a_pack_size;
  matmul->matrix_b_.pack_size_ = b_pack_size;
  compute->row_align_ = UP_ROUND(compute->row_, compute->row_tile_);
  matmul->out_need_aligned_ = (matmul->out_need_aligned_ && ((compute->col_ % compute->col_tile_) != 0));
  compute->col_step_ = matmul->out_need_aligned_ ? compute->col_align_ : compute->col_;
  NNACL_CHECK_FALSE(INT_MUL_OVERFLOW(matmul->a_batch_, compute->row_), NNACL_ERR);
  compute->row_num_ = matmul->a_batch_ * compute->row_;
  return NNACL_OK;
}

int MatmulBasePackMatrixAImplOpt(MatmulStruct *matmul) { return NNACL_ERR; }

int MatmulBasePackMatrixAImpl(MatmulStruct *matmul) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  float *src_ptr = (matmul->matrix_a_.origin_ptr_ != NULL) ? (matmul->matrix_a_.origin_ptr_)
                                                           : (float *)(matmul->base_.in_[FIRST_INPUT]->data_);
  NNACL_CHECK_TRUE_RET(src_ptr != NULL, NNACL_ERR);
  NNACL_CHECK_TRUE_RET(matmul->matrix_a_.pack_ptr_ != NULL, NNACL_ERR);
  NNACL_CHECK_TRUE_RET(matmul->matrix_a_pack_fun_ != NULL, NNACL_ERR);
  for (int i = 0; i < matmul->a_batch_; i++) {
    const float *src = src_ptr + i * matmul->compute_.deep_ * matmul->compute_.row_;
    float *dst = matmul->matrix_a_.pack_ptr_ + i * matmul->compute_.deep_ * matmul->compute_.row_align_;
    if (param->a_transpose_) {
      matmul->matrix_a_pack_fun_(src, dst, matmul->compute_.deep_, matmul->compute_.row_, 0, matmul->compute_.deep_);
    } else {
      matmul->matrix_a_pack_fun_(src, dst, matmul->compute_.row_, matmul->compute_.deep_, 0, matmul->compute_.row_);
    }
  }
  return NNACL_OK;
}

int MatmulBasePackMatrixBImpl(MatmulStruct *matmul) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);

  float *src_ptr = matmul->matrix_b_.origin_ptr_ != NULL ? matmul->matrix_b_.origin_ptr_
                                                         : (float *)matmul->base_.in_[SECOND_INPUT]->data_;
  NNACL_CHECK_TRUE_RET(src_ptr != NULL, NNACL_ERR);
  NNACL_CHECK_TRUE_RET(matmul->matrix_b_.pack_ptr_ != NULL, NNACL_ERR);
  NNACL_CHECK_TRUE_RET(matmul->matrix_b_pack_fun_ != NULL, NNACL_ERR);

  for (int i = 0; i < matmul->b_batch_; i++) {
    if (param->b_transpose_) {
      matmul->compute_.pack_b_stride_ = UP_DIV(matmul->compute_.col_, matmul->base_.thread_nr_);
    } else {
      matmul->compute_.pack_b_stride_ = UP_DIV(matmul->compute_.deep_, matmul->base_.thread_nr_);
    }
    matmul->pack_b_src_ = src_ptr + i * matmul->compute_.deep_ * matmul->compute_.col_;
    matmul->pack_b_dst_ = matmul->matrix_b_.pack_ptr_ + i * matmul->compute_.deep_ * matmul->compute_.col_align_;
    int ret = matmul->base_.env_->ParallelLaunch(matmul->base_.env_->thread_pool_, MatmulFp32PackMatrixBRun, matmul,
                                                 matmul->base_.thread_nr_);
    NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
  }
  return NNACL_OK;
}

int MatmulBasePackMatrixA(MatmulStruct *matmul) {
  if (!matmul->a_const_) {
    if (!matmul->matrix_a_.need_pack_) {
      matmul->matrix_a_.pack_ptr_ = (float *)matmul->base_.in_[0]->data_;
      return NNACL_OK;
    }
    if (matmul->base_.train_session_) {
      matmul->matrix_a_.pack_ptr_ = (float *)(matmul->base_.workspace_);
    } else {
      matmul->matrix_a_.pack_ptr_ = (float *)(matmul->base_.env_->Alloc(matmul->base_.env_->allocator_,
                                                                        matmul->matrix_a_.pack_size_ * sizeof(float)));
    }
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(matmul->matrix_a_.pack_ptr_);
  } else {
    bool is_packed = false;
    void *data = NULL;
    size_t data_size = (size_t)(matmul->matrix_a_.pack_size_) * sizeof(float);
    if (matmul->is_sharing_pack_) {
      TensorC *a_matrix = matmul->base_.in_[FIRST_INPUT];
      data = matmul->get_sharing_weight_(matmul->shaing_manager_, a_matrix->data_, data_size, &is_packed);
    } else {
      data = malloc(data_size);
    }
    matmul->matrix_a_.pack_ptr_ = (float *)data;
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(matmul->matrix_a_.pack_ptr_);
    if (is_packed) {
      return NNACL_OK;
    }
  }
  if (matmul->pack_opt_) {
    /* valid in arm64 */
    return matmul->pack_matrix_a_impl_opt_(matmul);
  }
  return matmul->pack_matrix_a_impl_(matmul);
}

int MatmulBasePackMatrixB(MatmulStruct *matmul) {
  if (!matmul->b_const_) {
    if (!matmul->matrix_b_.need_pack_) {
      matmul->matrix_b_.pack_ptr_ = (float *)matmul->base_.in_[SECOND_INPUT]->data_;
      return NNACL_OK;
    }
    if (matmul->base_.train_session_) {
      matmul->matrix_b_.pack_ptr_ = (float *)(matmul->base_.workspace_) + matmul->matrix_a_.pack_size_;
    } else {
      matmul->matrix_b_.pack_ptr_ = (float *)(matmul->base_.env_->Alloc(matmul->base_.env_->allocator_,
                                                                        matmul->matrix_b_.pack_size_ * sizeof(float)));
    }
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(matmul->matrix_b_.pack_ptr_);
  } else {
    if (!matmul->matrix_b_.need_pack_ && matmul->weight_is_packed_) {
      matmul->matrix_b_.pack_ptr_ = (float *)matmul->base_.in_[SECOND_INPUT]->data_;
      return NNACL_OK;
    }
    bool is_packed = false;
    void *data = NULL;
    size_t data_size = (size_t)(matmul->matrix_b_.pack_size_) * sizeof(float);
    if (matmul->is_sharing_pack_) {
      TensorC *b_matrix = matmul->base_.in_[SECOND_INPUT];
      data = matmul->get_sharing_weight_(matmul->shaing_manager_, b_matrix->data_, data_size, &is_packed);
    } else {
      data = malloc(data_size);
    }
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(data);
    matmul->matrix_b_.pack_ptr_ = (float *)data;
    if (is_packed) {
      return NNACL_OK;
    }
  }
  return matmul->pack_matrix_b_impl_(matmul);
}

int MatmulBaseBackupConstMatrix(MatmulStruct *matmul, MatrixInfo *matrix_info, int index) {
  NNACL_CHECK_TRUE_RET(index < (int)matmul->base_.in_size_, NNACL_ERR);
  size_t backup_size = (size_t)GetElementNum(matmul->base_.in_[index]) * sizeof(float);
  NNACL_CHECK_TRUE_RET(backup_size > 0, NNACL_ERR);
  matrix_info->origin_ptr_ = (float *)(matmul->base_.env_->Alloc(matmul->base_.env_->allocator_, backup_size));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(matrix_info->origin_ptr_);
  void *src_ptr = matmul->base_.in_[index]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(src_ptr);
  (void)memcpy(matrix_info->origin_ptr_, src_ptr, backup_size);
  matrix_info->origin_need_free_ = true;
  return NNACL_OK;
}

int MatmulBaseParallelRunByRow(MatmulStruct *matmul, int task_id) { return NNACL_ERR; }

int MatmulBaseParallelRunByBatch(MatmulStruct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MatmulComputeParam *compute = &matmul->compute_;

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
      MatMulOpt(a, b, c, bias, param->act_type_, compute->deep_, compute->row_, compute->col_step_, compute->col_,
                OutType_Nhwc);
    } else if (func_flag == C1NUM) {
      MatVecMulFp32Block8(a, b, c, bias, param->act_type_, compute->deep_, compute->col_step_);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, compute->deep_, compute->col_step_, compute->col_step_);
    }
  }
  return NNACL_OK;
}

int MatmulBaseParallelRunIsNotPackByBatch(MatmulStruct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  int start_batch = task_id * matmul->compute_.batch_stride_;
  int end_batch = MSMIN(matmul->batch_, start_batch + matmul->compute_.batch_stride_);
  float bias = 0;
  if (matmul->matrix_c_.pack_ptr_ != NULL) {
    bias = matmul->matrix_c_.pack_ptr_[0];
  }
  for (int index = start_batch; index < end_batch; ++index) {
    float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * matmul->compute_.row_ * matmul->compute_.deep_;
    float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * matmul->compute_.deep_ * matmul->compute_.col_;
    float *c = matmul->output_data_ + index * matmul->compute_.row_ * matmul->compute_.col_;
    matmul->gemm_not_pack_fun_(a, b, c, &bias, matmul->compute_.row_, matmul->compute_.deep_, param->act_type_);
  }
  return NNACL_OK;
}

void MatmulBaseGetThreadCuttingInfoByRow(MatmulStruct *matmul) {
  int row_step = NNACL_MAX(matmul->compute_.row_num_ / matmul->base_.thread_nr_, matmul->compute_.row_min_unit_);
  int row_remaining = matmul->compute_.row_num_ - row_step * matmul->base_.thread_nr_;

  int split_point = 0;
  int count = 0;
  while (split_point < matmul->compute_.row_num_) {
    matmul->split_points_[count++] = split_point;
    split_point += row_step;
    if (row_remaining > 0) {
      ++split_point;
      --row_remaining;
    }
  }
  matmul->base_.thread_nr_ = count;
}

int MatmulBaseParallelRunByOC(MatmulStruct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  NNACL_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);
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
      MatVecMulFp32Block8(a, b, c, bias, act, compute->deep_, compute_oc);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, act, compute->deep_, compute_oc, compute->col_step_);
    }
  }
  return NNACL_OK;
}

void MatmulBaseGetThreadCuttingPolicy(MatmulStruct *matmul) {
  if (matmul->compute_.deep_ < kNumDeepThreshold) {
    if (matmul->model_thread_nr_ != -1) {
      matmul->base_.thread_nr_ = matmul->model_thread_nr_;
    }
  }

  if ((matmul->a_batch_ >= matmul->base_.thread_nr_ &&
       (matmul->b_batch_ == matmul->a_batch_ || !matmul->support_mul_batch_cut_by_row_)) ||
      matmul->compute_.col_ == 1) {
    matmul->compute_.batch_stride_ = UP_DIV(matmul->batch_, matmul->base_.thread_nr_);
    matmul->parallel_run_ = matmul->parallel_run_by_batch_;
    if (matmul->compute_.col_ != 1 || matmul->a_const_) {
      return;
    }

    matmul->parallel_run_ = matmul->parallel_run_not_pack_by_batch_;
    if (matmul->compute_.deep_ == 1) {
      matmul->gemm_not_pack_fun_ = GemmIsNotPack;
    } else {
      matmul->gemm_not_pack_fun_ = GemmIsNotPackOptimize;
      if (matmul->check_thread_cutting_by_row_(matmul)) {
        matmul->parallel_run_ = matmul->parallel_run_by_row_;
        matmul->get_thread_cutting_info_by_row_(matmul);
      }
    }
    return;
  } else if ((matmul->a_batch_ >= matmul->base_.thread_nr_ && matmul->b_batch_ == 1) ||
             matmul->check_thread_cutting_by_row_(matmul)) {
    matmul->parallel_run_ = matmul->parallel_run_by_row_;
    matmul->get_thread_cutting_info_by_row_(matmul);
  } else {
    int total_col_unit = UP_DIV(matmul->compute_.col_align_, matmul->compute_.col_min_unit_);
    matmul->base_.thread_nr_ = MSMIN(matmul->base_.thread_nr_, total_col_unit);
    int block_col_unit = UP_DIV(total_col_unit, matmul->base_.thread_nr_);

    int count = 0;
    int split_point = 0;
    while (split_point < total_col_unit) {
      matmul->split_points_[count++] = (split_point * matmul->compute_.col_min_unit_);
      split_point += block_col_unit;
    }
    matmul->base_.thread_nr_ = count;
    matmul->parallel_run_ = matmul->parallel_run_by_oc_;
  }
  return;
}

int MatmulBasePackBiasMatrix(MatmulStruct *matmul) {
  if (matmul->base_.in_size_ != FOURTH_INPUT) {
    return NNACL_OK;
  }
  if (matmul->matrix_c_.has_packed_) {
    NNACL_CHECK_FALSE(matmul->matrix_c_.pack_size_ < matmul->compute_.col_align_, NNACL_ERR);
    return NNACL_OK;
  }
  TensorC *bias_tensor = matmul->base_.in_[THIRD_INPUT];
  float *bias_src = matmul->matrix_c_.origin_ptr_ != NULL ? matmul->matrix_c_.origin_ptr_ : (float *)bias_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(bias_src);

  int bias_num = GetElementNum(bias_tensor);
  NNACL_CHECK_TRUE_RET(bias_num > 0 && matmul->compute_.col_align_ >= bias_num, NNACL_ERR);

  matmul->matrix_c_.pack_size_ = matmul->compute_.col_align_;
  if (matmul->matrix_c_.pack_ptr_ == NULL) {
    matmul->matrix_c_.pack_ptr_ = (float *)(malloc(matmul->matrix_c_.pack_size_ * sizeof(float)));
  }
  NNACL_CHECK_NULL_RETURN_ERR(matmul->matrix_c_.pack_ptr_);

  if (bias_num == 1) {
    for (int i = 0; i < matmul->matrix_c_.pack_size_; ++i) {
      matmul->matrix_c_.pack_ptr_[i] = bias_src[0];
    }
  } else {
    (void)memcpy(matmul->matrix_c_.pack_ptr_, bias_src, bias_num * sizeof(float));
    (void)memset(matmul->matrix_c_.pack_ptr_ + bias_num, 0, (matmul->matrix_c_.pack_size_ - bias_num) * sizeof(float));
  }
  if (matmul->matrix_c_.origin_need_free_) {
    matmul->base_.env_->Free(matmul->base_.env_->allocator_, matmul->matrix_c_.origin_ptr_);
    matmul->matrix_c_.origin_ptr_ = NULL;
    matmul->matrix_c_.origin_need_free_ = false;
  }
  return NNACL_OK;
}

int MatmulBaseInitTmpOutBuffer(MatmulStruct *matmul) {
  if (matmul->out_need_aligned_) {
    if (matmul->output_data_ != NULL) {
      matmul->base_.env_->Free(matmul->base_.env_->allocator_, matmul->output_data_);
    }
    // avx need to malloc dst aligned to C8NUM
    // avx512 need to malloc dst aligned to C16NUM
    int out_channel = matmul->compute_.col_;
    NNACL_CHECK_ZERO_RETURN_ERR(matmul->compute_.col_tile_);
    int oc_block_num = UP_DIV(out_channel, matmul->compute_.col_tile_);
    int ele_num = matmul->batch_ * matmul->compute_.row_ * oc_block_num * matmul->compute_.col_tile_;
    int data_size = ele_num * (int)sizeof(float);
    matmul->output_data_ = (float *)(matmul->base_.env_->Alloc(matmul->base_.env_->allocator_, data_size));
    NNACL_CHECK_NULL_RETURN_ERR(matmul->output_data_);
  }
  return NNACL_OK;
}

void MatmulBaseInitGlobalVariable(MatmulStruct *matmul) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  matmul->matrix_a_.need_pack_ = true;
  matmul->matrix_b_.need_pack_ = !matmul->weight_is_packed_;
  matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2Row12MajorParallel : RowMajor2Col12MajorParallel;
  matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2Col8MajorParallel : RowMajor2Row8MajorParallel;
  matmul->compute_.row_tile_ = C12NUM;
  matmul->compute_.col_tile_ = C8NUM;
  matmul->compute_.col_min_unit_ = C8NUM;
  return;
}

bool MatmulBaseCheckThreadCuttingByRow() { return false; }

void MatmulBaseFreePackedMatrixA(KernelBase *self) {
  MatmulStruct *matmul = (MatmulStruct *)self;
  if (matmul->matrix_a_.need_pack_ && !matmul->base_.train_session_ && matmul->matrix_a_.pack_ptr_ != NULL) {
    self->env_->Free(self->env_->allocator_, matmul->matrix_a_.pack_ptr_);
  }
  matmul->matrix_a_.pack_ptr_ = NULL;
}

void MatmulBaseFreePackedMatrixB(KernelBase *self) {
  MatmulStruct *matmul = (MatmulStruct *)self;
  if (matmul->matrix_b_.need_pack_ && !matmul->base_.train_session_ && matmul->matrix_b_.pack_ptr_ != NULL) {
    matmul->base_.env_->Free(matmul->base_.env_->allocator_, matmul->matrix_b_.pack_ptr_);
  }
  matmul->matrix_b_.pack_ptr_ = NULL;
}

int MatmulBaseResize(KernelBase *self) {
  MatmulStruct *matmul = (MatmulStruct *)self;

  int ret = matmul->init_parameter_(matmul);
  NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
  if (self->train_session_) {
    self->work_size_ = (matmul->matrix_a_.pack_size_ + matmul->matrix_b_.pack_size_) * (int)sizeof(float);
  }

  matmul->get_thread_cutting_policy_(matmul);
  if (!matmul->matrix_c_.has_packed_) {
    ret = MatmulBasePackBiasMatrix(matmul);
    NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
    if (!matmul->bias_need_repack_) {
      matmul->matrix_c_.has_packed_ = true;
    }
  }
  ret = MatmulBaseInitTmpOutBuffer(matmul);
  NNACL_CHECK_FALSE(ret != NNACL_OK, ret);

  return NNACL_OK;
}

int MatmulBaseRelease(struct KernelBase *self) {
  MatmulStruct *matmul = (MatmulStruct *)self;
  MatmulBaseFreeBatchOffset(matmul);

  if (matmul->out_need_aligned_ && matmul->output_data_ != NULL) {
    matmul->base_.env_->Free(matmul->base_.env_->allocator_, matmul->output_data_);
    matmul->output_data_ = NULL;
  }
  if (matmul->matrix_c_.pack_ptr_ != NULL) {
    free(matmul->matrix_c_.pack_ptr_);
    matmul->matrix_c_.pack_ptr_ = NULL;
  }
  if (matmul->a_const_) {
    if (matmul->is_sharing_pack_) {
      matmul->free_sharing_weight_(matmul->shaing_manager_, matmul->matrix_a_.pack_ptr_);
    } else {
      free(matmul->matrix_a_.pack_ptr_);
    }
  }
  if (matmul->b_const_) {
    if (!matmul->matrix_b_.need_pack_ && matmul->weight_is_packed_) {
      return NNACL_OK;
    }
    if (matmul->is_sharing_pack_) {
      matmul->free_sharing_weight_(matmul->shaing_manager_, matmul->matrix_b_.pack_ptr_);
    } else {
      free(matmul->matrix_b_.pack_ptr_);
    }
  }
  return NNACL_OK;
}

int MatmulBasePrepare(struct KernelBase *self) {
  MatmulStruct *matmul = (MatmulStruct *)self;

  NNACL_CHECK_FALSE(matmul->base_.in_size_ < C2NUM, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(matmul->base_.out_size_ < 1, NNACL_OUTPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(matmul->base_.in_[FIRST_INPUT]->data_type_ != kNumberTypeFloat32, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(matmul->base_.in_[SECOND_INPUT]->data_type_ != kNumberTypeFloat32, NNACL_INPUT_TENSOR_ERROR);

  if (matmul->base_.in_size_ == THREE_TENSOR) {
    NNACL_CHECK_TRUE_RET(matmul->base_.in_[THIRD_INPUT]->data_type_ == kNumberTypeFloat32, NNACL_MATMUL_BIAS_INVALID);
  }

  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  NNACL_CHECK_FALSE(
    param->act_type_ != ActType_No && param->act_type_ != ActType_Relu && param->act_type_ != ActType_Relu6,
    NNACL_MATMUL_ACT_TYPE_INVALID);

  int ret = matmul->init_parameter_(matmul);
  NNACL_CHECK_FALSE(ret != NNACL_OK, ret);

  if (matmul->a_const_) {
    ret = MatmulBasePackMatrixA(matmul);
    NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
    matmul->matrix_a_.has_packed_ = true;
  }
  if (matmul->b_const_) {
    ret = MatmulBasePackMatrixB(matmul);
    NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
    matmul->matrix_b_.has_packed_ = true;
  }

  if (matmul->base_.in_size_ == THREE_TENSOR) {
    /* deal with const bias */
    bool bias_const = IsConst(self->in_[THIRD_INPUT]);
    if (!matmul->infer_shape_ && bias_const && !matmul->base_.train_session_ && matmul->matrix_c_.origin_ptr_ == NULL) {
      ret = MatmulBaseBackupConstMatrix(matmul, &matmul->matrix_c_, THIRD_INPUT);
      NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
    }
  }
  return NNACL_OK;
}

int MatmulBaseCompute(struct KernelBase *self) {
  MatmulStruct *matmul = (MatmulStruct *)self;

  float *out_data = (float *)(matmul->base_.out_[FIRST_INPUT]->data_);
  NNACL_CHECK_FALSE(out_data == NULL, NNACL_ERR);
  if (!matmul->out_need_aligned_) {
    matmul->output_data_ = out_data;
  }

  if (!matmul->a_const_) {
    int ret = MatmulBasePackMatrixA(matmul);
    NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
  }
  if (!matmul->b_const_) {
    int ret = MatmulBasePackMatrixB(matmul);
    NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
  }

  NNACL_CHECK_NULL_RETURN_ERR(matmul->matrix_a_.pack_ptr_);
  NNACL_CHECK_NULL_RETURN_ERR(matmul->matrix_b_.pack_ptr_);

  int ret = self->env_->ParallelLaunch(self->env_->thread_pool_, MatmulFp32Run, self, self->thread_nr_);
  NNACL_CHECK_FALSE(ret != NNACL_OK, ret);

  if (matmul->out_need_aligned_) {
    PackNHWCXToNHWCFp32(matmul->output_data_, out_data, matmul->batch_, matmul->compute_.row_, matmul->compute_.col_,
                        matmul->compute_.col_tile_);
  } else {
    matmul->output_data_ = NULL;
  }
  if (!matmul->a_const_) {
    MatmulBaseFreePackedMatrixA(self);
  }

  if (!matmul->b_const_) {
    MatmulBaseFreePackedMatrixB(self);
  }
  return NNACL_OK;
}

void InitMatrixInfo(MatrixInfo *info) {
  info->need_pack_ = false;
  info->has_packed_ = false;
  info->origin_need_free_ = false;
  info->pack_size_ = -1;
  info->origin_ptr_ = NULL;
  info->pack_ptr_ = NULL;
}

KernelBase *CreateMatmulBase() {
  MatmulStruct *matmul = (MatmulStruct *)malloc(sizeof(MatmulStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(matmul);
  memset(matmul, 0, sizeof(MatmulStruct));
  matmul->base_.Prepare = MatmulBasePrepare;
  matmul->base_.Resize = MatmulBaseResize;
  matmul->base_.Release = MatmulBaseRelease;
  matmul->base_.Compute = MatmulBaseCompute;
  InitMatrixInfo(&(matmul->matrix_a_));
  InitMatrixInfo(&(matmul->matrix_b_));
  InitMatrixInfo(&(matmul->matrix_c_));
  matmul->is_sharing_pack_ = false;
  matmul->pack_opt_ = false;
  matmul->a_const_ = false;
  matmul->b_const_ = false;
  matmul->bias_need_repack_ = false;
  matmul->out_need_aligned_ = false;
  matmul->a_offset_ = NULL;
  matmul->b_offset_ = NULL;
  matmul->model_thread_nr_ = -1;
  matmul->support_mul_batch_cut_by_row_ = false;
  matmul->matmul_type_ = kMatmulFp32BaseCpu;
  matmul->get_thread_cutting_policy_ = MatmulBaseGetThreadCuttingPolicy;
  matmul->check_thread_cutting_by_row_ = MatmulBaseCheckThreadCuttingByRow;
  matmul->get_thread_cutting_info_by_row_ = MatmulBaseGetThreadCuttingInfoByRow;
  matmul->init_parameter_ = MatmulBaseInitParameter;
  matmul->init_global_varibale_ = MatmulBaseInitGlobalVariable;
  matmul->pack_matrix_a_impl_opt_ = MatmulBasePackMatrixAImplOpt;
  matmul->pack_matrix_a_impl_ = MatmulBasePackMatrixAImpl;
  matmul->pack_matrix_b_impl_ = MatmulBasePackMatrixBImpl;
  matmul->parallel_run_by_batch_ = MatmulBaseParallelRunByBatch;
  matmul->parallel_run_not_pack_by_batch_ = MatmulBaseParallelRunIsNotPackByBatch;
  matmul->parallel_run_by_oc_ = MatmulBaseParallelRunByOC;
  matmul->parallel_run_by_row_ = MatmulBaseParallelRunByRow;
  return (KernelBase *)matmul;
}
