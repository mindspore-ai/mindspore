/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/cpu/fp32/matmul_fp32_base.h"
#include <algorithm>
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "src/runtime/kernel/cpu/fp32/matmul_fp32_arm32.h"
#include "src/runtime/kernel/cpu/fp32/matmul_fp32_arm64.h"
#include "src/runtime/kernel/cpu/fp32/matmul_fp32_sse.h"
#include "src/runtime/kernel/cpu/fp32/matmul_fp32_avx.h"
#include "src/runtime/kernel/cpu/fp32/matmul_fp32_avx512.h"
#include "src/runtime/kernel/cpu/fp32/matmul_fp32_common.h"
#include "nnacl/fp32/pack_fp32_opt.h"

using mindspore::lite::RET_NULL_PTR;

namespace mindspore::kernel {
int MatmulRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<const MatmulFp32BaseCPUKernel *>(cdata);
  auto error_code = (op->*(op->parallel_fun_))(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "MatmulRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

MatmulFp32BaseCPUKernel::~MatmulFp32BaseCPUKernel() {
  // packed const-matrix will be delete by framework.
  if (out_need_aligned_ && output_data_ != nullptr) {
    free(output_data_);
    output_data_ = nullptr;
  }
  if (matrix_c_.pack_ptr != nullptr) {
    free(matrix_c_.pack_ptr);
    matrix_c_.pack_ptr = nullptr;
  }
  if (params_->a_const_) {
    lite::PackWeightManager::GetInstance()->Free(matrix_a_.pack_ptr);
  }
  if (params_->b_const_) {
    lite::PackWeightManager::GetInstance()->Free(matrix_b_.pack_ptr);
  }
}

int MatmulFp32BaseCPUKernel::BackupConstMatrix(MatrixInfo *matrix_info, int index) {
  MS_CHECK_TRUE_MSG(index < static_cast<int>(in_tensors_.size()), RET_ERROR, "matrix is not existing.");
  auto element_num = in_tensors_[index]->ElementsNum();
  MS_CHECK_TRUE_MSG(element_num > 0, RET_ERROR, "matrix is invalid.");
  matrix_info->origin_ptr = reinterpret_cast<float *>(ms_context_->allocator->Malloc(element_num * sizeof(float)));
  MS_CHECK_TRUE_MSG(matrix_info->origin_ptr != nullptr, RET_ERROR, "matrix is invalid.");
  auto src_ptr = in_tensors_[index]->data();
  MS_CHECK_TRUE_MSG(src_ptr != nullptr, RET_ERROR, "matrix is invalid.");
  memcpy(matrix_info->origin_ptr, src_ptr, element_num * sizeof(float));
  matrix_info->has_origin = true;
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::PackMatrixA() {
  if (!params_->a_const_) {
    if (!matrix_a_.need_pack) {
      matrix_a_.pack_ptr = reinterpret_cast<float *>(in_tensors_[FIRST_INPUT]->data());
      return RET_OK;
    }
    if (op_parameter_->is_train_session_) {
      matrix_a_.pack_ptr = reinterpret_cast<float *>(workspace());
    } else {
      matrix_a_.pack_ptr =
        reinterpret_cast<float *>(ms_context_->allocator->Malloc(matrix_a_.pack_size * sizeof(float)));
    }
  } else {
    bool is_packed = false;
    void *data = lite::PackWeightManager::GetInstance()->GetPackData(
      in_tensors()[FIRST_INPUT]->data(), static_cast<size_t>(matrix_a_.pack_size) * sizeof(float), &is_packed);
    matrix_a_.pack_ptr = reinterpret_cast<float *>(data);
    if (matrix_a_.pack_ptr == nullptr) {
      MS_LOG(ERROR) << "matrix a pack ptr is nullptr.";
      return RET_ERROR;
    }
    if (is_packed) {
      return RET_OK;
    }
  }
  if (pack_opt_) {
    return PackMatrixAImplOpt();  // currently, only arm64 support.
  }
  return PackMatrixAImpl();
}

int MatmulFp32BaseCPUKernel::PackMatrixAImpl() {
  auto src_ptr =
    matrix_a_.has_origin ? matrix_a_.origin_ptr : reinterpret_cast<float *>(in_tensors_[FIRST_INPUT]->data());
  MS_CHECK_TRUE_MSG(src_ptr != nullptr, RET_ERROR, "matrix-a source ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_a_.pack_ptr != nullptr, RET_ERROR, "matrix-a pack ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_a_pack_fun_ != nullptr, RET_ERROR, "matrix-a func is a nullptr.");
  for (int i = 0; i < a_batch_; i++) {
    const float *src = src_ptr + i * params_->deep_ * params_->row_;
    float *dst = matrix_a_.pack_ptr + i * params_->deep_ * params_->row_align_;
    if (params_->a_transpose_) {
      matrix_a_pack_fun_(src, dst, params_->deep_, params_->row_);
    } else {
      matrix_a_pack_fun_(src, dst, params_->row_, params_->deep_);
    }
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::FreePackedMatrixA() {
  if (matrix_a_.need_pack && !op_parameter_->is_train_session_ && matrix_a_.pack_ptr != nullptr) {
    ms_context_->allocator->Free(matrix_a_.pack_ptr);
  }
  matrix_a_.pack_ptr = nullptr;
}

int MatmulFp32BaseCPUKernel::PackMatrixB() {
  if (!params_->b_const_) {
    if (!matrix_b_.need_pack) {
      matrix_b_.pack_ptr = reinterpret_cast<float *>(in_tensors_[SECOND_INPUT]->data());
      return RET_OK;
    }
    if (op_parameter_->is_train_session_) {
      matrix_b_.pack_ptr = reinterpret_cast<float *>(workspace()) + matrix_a_.pack_size;
    } else {
      matrix_b_.pack_ptr =
        reinterpret_cast<float *>(ms_context_->allocator->Malloc(matrix_b_.pack_size * sizeof(float)));
    }
  } else {
    bool is_packed = false;
    void *data = lite::PackWeightManager::GetInstance()->GetPackData(
      in_tensors()[SECOND_INPUT]->data(), static_cast<size_t>(matrix_b_.pack_size) * sizeof(float), &is_packed);
    matrix_b_.pack_ptr = reinterpret_cast<float *>(data);
    if (matrix_b_.pack_ptr == nullptr) {
      MS_LOG(ERROR) << "matrix b pack ptr is nullptr.";
      return RET_ERROR;
    }
    if (is_packed) {
      return RET_OK;
    }
  }
  return PackMatrixBImpl();
}

int MatmulFp32BaseCPUKernel::PackMatrixBImpl() {
  auto src_ptr =
    matrix_b_.has_origin ? matrix_b_.origin_ptr : reinterpret_cast<float *>(in_tensors_[SECOND_INPUT]->data());
  MS_CHECK_TRUE_MSG(src_ptr != nullptr, RET_ERROR, "matrix-b source ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_b_.pack_ptr != nullptr, RET_ERROR, "matrix-b pack ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_b_pack_fun_ != nullptr, RET_ERROR, "matrix-b func is a nullptr.");
  for (int i = 0; i < b_batch_; i++) {
    const float *src = src_ptr + i * params_->deep_ * params_->col_;
    float *dst = matrix_b_.pack_ptr + i * params_->deep_ * params_->col_align_;
    if (params_->b_transpose_) {
      matrix_b_pack_fun_(src, dst, params_->col_, params_->deep_);
    } else {
      matrix_b_pack_fun_(src, dst, params_->deep_, params_->col_);
    }
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::FreePackedMatrixB() {
  if (matrix_b_.need_pack && !op_parameter_->is_train_session_ && matrix_b_.pack_ptr != nullptr) {
    ms_context_->allocator->Free(matrix_b_.pack_ptr);
  }
  matrix_b_.pack_ptr = nullptr;
}

int MatmulFp32BaseCPUKernel::PackBiasMatrix() {
  if (in_tensors_.size() != FOURTH_INPUT) {
    return RET_OK;
  }
  if (matrix_c_.has_packed) {
    if (matrix_c_.pack_size < params_->col_align_) {
      MS_LOG(ERROR) << "matmul don't support that column is dynamic.";
      return RET_ERROR;
    }
    return RET_OK;
  }
  auto bias_tensor = in_tensors_[THIRD_INPUT];
  if (bias_tensor == nullptr) {
    MS_LOG(ERROR) << "bias_tensor invalid";
    return RET_ERROR;
  }
  auto bias_src = matrix_c_.has_origin ? matrix_c_.origin_ptr : reinterpret_cast<float *>(bias_tensor->data());
  MS_CHECK_TRUE_MSG(bias_src != nullptr, RET_ERROR, "matrix-c is a nullptr.");
  auto bias_num = bias_tensor->ElementsNum();
  MS_CHECK_TRUE_MSG(bias_num > 0 && params_->col_align_ >= bias_num, RET_ERROR, "matrix-c is invalid.");
  matrix_c_.pack_size = params_->col_align_;
  matrix_c_.pack_ptr = reinterpret_cast<float *>(malloc(static_cast<size_t>(matrix_c_.pack_size) * sizeof(float)));
  MS_CHECK_TRUE_MSG(matrix_c_.pack_ptr != nullptr, RET_ERROR, "matrix-c malloc failed.");
  if (bias_num == 1) {
    for (int i = 0; i < matrix_c_.pack_size; ++i) {
      matrix_c_.pack_ptr[i] = bias_src[0];
    }
  } else {
    (void)memcpy(matrix_c_.pack_ptr, bias_src, bias_num * static_cast<int>(sizeof(float)));
    memset(matrix_c_.pack_ptr + bias_num, 0, (matrix_c_.pack_size - bias_num) * sizeof(float));
  }
  if (matrix_c_.has_origin) {
    ms_context_->allocator->Free(matrix_c_.origin_ptr);
    matrix_c_.origin_ptr = nullptr;
    matrix_c_.has_origin = false;
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::ParallelRunIsNotPackByBatch(int task_id) const {
  int start_batch = task_id * batch_stride_;
  int end_batch = MSMIN(params_->batch, start_batch + batch_stride_);
  float bias = 0;
  if (matrix_c_.pack_ptr != nullptr) {
    bias = matrix_c_.pack_ptr[0];
  }
  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * params_->row_ * params_->deep_;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * params_->deep_ * params_->col_;
    float *c = output_data_ + index * params_->row_ * params_->col_;
    gemmIsNotPackFun(a, b, c, &bias, params_->row_, params_->deep_);
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  MS_CHECK_TRUE_MSG(in_tensors_[FIRST_INPUT]->data_type() == kNumberTypeFloat32, RET_ERROR,
                    "matrix-a's data type is invalid.");
  MS_CHECK_TRUE_MSG(in_tensors_[SECOND_INPUT]->data_type() == kNumberTypeFloat32, RET_ERROR,
                    "matrix-b's data type is invalid.");
  if (in_tensors_.size() == FOURTH_INPUT) {
    MS_CHECK_TRUE_MSG(in_tensors_[THIRD_INPUT]->IsConst(), RET_ERROR, "matrix-c must be const when existing.");
    MS_CHECK_TRUE_MSG(in_tensors_[THIRD_INPUT]->data_type() == kNumberTypeFloat32, RET_ERROR,
                      "matrix-c's data type is invalid.");
  }
  auto ret = InitParameter();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Init parameters failed.");
  if (params_->a_const_) {
    ret = PackMatrixA();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "pack const-matrix a failed.");
    matrix_a_.has_packed = true;
  }
  if (params_->b_const_) {
    ret = PackMatrixB();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "pack const-matrix b failed.");
    matrix_b_.has_packed = true;
  }
  if (!InferShapeDone()) {
    if (in_tensors_.size() == FOURTH_INPUT && !op_parameter_->is_train_session_) {
      ret = BackupConstMatrix(&matrix_c_, THIRD_INPUT);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "backup matrix-c failed.");
    }
    return RET_OK;
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::ReSize() {
  auto ret = InitParameter();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Init parameters failed.");
  if (op_parameter_->is_train_session_) {
    set_workspace_size((matrix_a_.pack_size + matrix_b_.pack_size) * static_cast<int>(sizeof(float)));
  }
  ret = GetThreadCuttingPolicy();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ThreadCuttingPolicy error!";
    return ret;
  }
  if (!matrix_c_.has_packed) {
    ret = PackBiasMatrix();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "pack const-matrix c failed.");
    matrix_c_.has_packed = true;
  }
  ret = InitTmpOutBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitTmpOutBuffer error!";
    return ret;
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitParameter() {
  InitGlobalVariable();
  if (params_->row_ == 1) {
    row_tile_ = 1;
    matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2ColMajor : RowMajor2RowMajor;
    matrix_a_.need_pack = false;
    pack_opt_ = false;
  }
  if (params_->col_ == 1 && !params_->a_const_) {
    out_need_aligned_ = false;
    row_tile_ = 1;
    col_tile_ = 1;
    matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2ColMajor : RowMajor2RowMajor;
    matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2ColMajor : RowMajor2RowMajor;
    matrix_a_.need_pack = params_->a_transpose_ && params_->row_ != 1;
    matrix_b_.need_pack = false;
    pack_opt_ = false;
  }
  params_->row_align_ = UP_ROUND(params_->row_, row_tile_);
  params_->col_align_ = UP_ROUND(params_->col_, col_tile_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_, params_->row_align_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_ * params_->row_align_, params_->deep_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_, params_->col_align_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_ * params_->col_align_, params_->deep_, RET_ERROR);
  auto a_pack_size = a_batch_ * params_->row_align_ * params_->deep_;
  auto b_pack_size = b_batch_ * params_->col_align_ * params_->deep_;
  if ((matrix_a_.has_packed && matrix_a_.pack_size != a_pack_size) ||
      (matrix_b_.has_packed && matrix_b_.pack_size != b_pack_size)) {
    MS_LOG(ERROR) << "matmul don't support dynamic packing if matrix is a constant.";
    return RET_ERROR;
  }
  matrix_a_.pack_size = a_pack_size;
  matrix_b_.pack_size = b_pack_size;
  params_->row_align_ = UP_ROUND(params_->row_, row_tile_);
  out_need_aligned_ = (out_need_aligned_ && ((params_->col_ % col_tile_) != 0));
  col_step_ = out_need_aligned_ ? params_->col_align_ : params_->col_;
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(a_batch_, params_->row_), RET_ERROR);
  row_num_ = a_batch_ * params_->row_;
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitTmpOutBuffer() {
  if (out_need_aligned_) {
    if (output_data_ != nullptr) {
      free(output_data_);
    }
    // avx need to malloc dst aligned to C8NUM
    // avx512 need to malloc dst aligned to C16NUM
    int out_channel = params_->col_;
    int oc_block_num = UP_DIV(out_channel, col_tile_);
    output_data_ = reinterpret_cast<float *>(
      malloc(params_->batch * params_->row_ * oc_block_num * col_tile_ * static_cast<int>(sizeof(float))));
    if (output_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc tmp output data failed.";
      return RET_NULL_PTR;
    }
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::GetThreadCuttingPolicy() {
  if (params_->batch >= op_parameter_->thread_num_ || params_->col_ == 1) {
    thread_count_ = op_parameter_->thread_num_;
    batch_stride_ = UP_DIV(params_->batch, thread_count_);
    parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByBatch;
    if (params_->col_ != 1 || params_->a_const_) {
      return RET_OK;
    }
    parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunIsNotPackByBatch;
    if (params_->deep_ == 1) {
      gemmIsNotPackFun = GemmIsNotPack;
    } else {
      gemmIsNotPackFun = GemmIsNotPackOptimize;
      if (CheckThreadCuttingByRow()) {
        parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByRow;
        GetThreadCuttingInfoByRow();
      }
    }
    return RET_OK;
  }
  if (CheckThreadCuttingByRow()) {
    parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByRow;
    GetThreadCuttingInfoByRow();
  } else {
    int total_col_unit = UP_DIV(params_->col_align_, col_min_unit_);
    thread_count_ = MSMIN(op_parameter_->thread_num_, total_col_unit);
    int block_col_unit = UP_DIV(total_col_unit, thread_count_);
    split_points_.clear();
    int split_point = 0;
    while (split_point < total_col_unit) {
      split_points_.push_back(split_point * col_min_unit_);
      split_point += block_col_unit;
    }
    thread_count_ = split_points_.size();
    parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByOC;
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::GetThreadCuttingInfoByRow() {
  int row_step = MSMAX(row_num_ / op_parameter_->thread_num_, row_min_unit_);
  int row_remaining = row_num_ - row_step * op_parameter_->thread_num_;
  split_points_.clear();
  int split_point = 0;
  while (split_point < row_num_) {
    split_points_.push_back(split_point);
    split_point += row_step;
    if (row_remaining > 0) {
      ++split_point;
      --row_remaining;
    }
  }
  thread_count_ = split_points_.size();
}

int MatmulFp32BaseCPUKernel::Run() {
  auto out_data = reinterpret_cast<float *>(out_tensors_.front()->data());
  CHECK_NULL_RETURN(out_data);
  if (!out_need_aligned_) {
    output_data_ = out_data;
  }
  if (!params_->a_const_) {
    auto ret = PackMatrixA();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "pack const-matrix a failed.");
  }
  if (!params_->b_const_) {
    auto ret = PackMatrixB();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "pack const-matrix b failed.");
  }
  MS_CHECK_TRUE_MSG(matrix_a_.pack_ptr != nullptr, RET_ERROR, "matrix-a pack ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_b_.pack_ptr != nullptr, RET_ERROR, "matrix-b pack ptr is a nullptr.");

  auto ret = ParallelLaunch(this->ms_context_, MatmulRun, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulRun failed in split by batch";
    return ret;
  }

  if (out_need_aligned_) {
    PackNHWCXToNHWCFp32(output_data_, out_data, params_->batch, params_->row_, params_->col_, col_tile_);
  } else {
    output_data_ = nullptr;
  }
  if (!params_->a_const_) {
    FreePackedMatrixA();
  }

  if (!params_->b_const_) {
    FreePackedMatrixB();
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
