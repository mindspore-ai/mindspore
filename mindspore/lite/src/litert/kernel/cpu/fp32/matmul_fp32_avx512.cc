#ifdef ENABLE_AVX512
/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/matmul_fp32_avx512.h"
#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"
#include "nnacl/fp32/matmul_avx512_fp32.h"
#include "nnacl/fp32/matmul_avx512_mask_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

namespace mindspore::kernel {
namespace {
size_t min_calc_cost_ = 1 * 6 * 64 * 64;
}

void MatmulFp32AVX512CPUKernel::InitGlobalVariable() {
  matrix_a_.need_pack = true;
  matrix_b_.need_pack = true;
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col64MajorParallel : RowMajor2Row64MajorParallel;
  matrix_a_.need_pack = params_->a_transpose_;
  row_tile_ = C1NUM;
  col_tile_ = C16NUM;
  col_min_unit_ = C64NUM;

  if (params_->row_ == 1) {
    if (!params_->b_const_ && params_->col_ <= C128NUM) {
      out_need_aligned_ = true;
    }
  } else if (params_->col_ == 1) {
    out_need_aligned_ = true;
  } else {
    out_need_aligned_ = false;
  }

  if (params_->deep_ >= C128NUM) {
    out_need_aligned_ = false;
  }
}

int MatmulFp32AVX512CPUKernel::InitParameter() {
  if (params_->deep_ < C128NUM) {
    return MatmulFp32BaseCPUKernel::InitParameter();
  }
  InitGlobalVariable();
  if (params_->col_ == 1 && !params_->a_const_) {
    out_need_aligned_ = false;
    row_tile_ = 1;
    col_tile_ = 1;
    matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matrix_a_.need_pack = params_->a_transpose_ && params_->row_ != 1;
    matrix_b_.need_pack = false;
    pack_opt_ = false;
  } else if (params_->row_ == 1 && !params_->b_const_) {
    out_need_aligned_ = false;
    row_tile_ = 1;
    col_tile_ = 1;
    matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matrix_a_.need_pack = false;
    matrix_b_.need_pack = params_->b_transpose_;
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

int MatmulFp32AVX512CPUKernel::PackMatrixAImplOpt() {
  MS_LOG(ERROR) << "Matmul: don't support optimized-packing, only support single-thread currently.";
  return RET_ERROR;
}

int MatmulFp32AVX512CPUKernel::ParallelRunByBatch(int task_id) const {
  int start_batch = task_id * batch_stride_;
  int end_batch = MSMIN(params_->batch, start_batch + batch_stride_);
  int func_flag{0};
  if (params_->row_ == 1) {
    func_flag += (!params_->b_const_ && params_->col_ <= C128NUM) ? C2NUM : C1NUM;
  }

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * params_->row_align_ * params_->deep_;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * params_->deep_ * params_->col_align_;
    float *c = output_data_ + index * params_->row_ * col_step_;

    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr;
    if (func_flag == 0) {
      if (out_need_aligned_) {
        MatMulAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, params_->col_align_,
                         params_->row_);
      } else {
        MatMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, params_->col_,
                             params_->row_);
      }
    } else if (func_flag == C1NUM) {
      if (out_need_aligned_) {
        MatVecMulAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, params_->col_align_);
      } else {
        MatVecMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, params_->col_);
      }
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, col_step_);
    }
  }
  return RET_OK;
}

int MatmulFp32AVX512CPUKernel::ParallelRunByRow(int task_id) const {
  if (task_id < 0 || task_id >= thread_num_) {
    MS_LOG(ERROR) << "task_id " << task_id << " is out of range, node is " << name_;
    return RET_ERROR;
  }
  int start_row = split_points_[task_id];
  int end_row = row_num_;
  if (task_id < (thread_num_ - 1)) {
    end_row = split_points_[task_id + 1];
  }
  int row_num = end_row - start_row;
  if (row_num <= 0) {
    return RET_OK;
  }
  const float *input = matrix_a_.pack_ptr + start_row * params_->deep_;
  float *output = output_data_ + start_row * col_step_;
  if (params_->col_ == 1) {
    float bias = 0;
    if (matrix_c_.pack_ptr != nullptr) {
      bias = matrix_c_.pack_ptr[0];
    }
    gemmIsNotPackFun(input, matrix_b_.pack_ptr, output, &bias, row_num, params_->deep_, params_->act_type_);
  } else {
    if (out_need_aligned_) {
      MatMulAvx512Fp32(input, matrix_b_.pack_ptr, output, matrix_c_.pack_ptr, params_->act_type_, params_->deep_,
                       params_->col_align_, params_->col_align_, row_num);
    } else {
      MatMulMaskAvx512Fp32(input, matrix_b_.pack_ptr, output, matrix_c_.pack_ptr, params_->act_type_, params_->deep_,
                           params_->col_, params_->col_, row_num);
    }
  }
  return RET_OK;
}

int MatmulFp32AVX512CPUKernel::ParallelRunByOC(int task_id) const {
  if (task_id < 0 || task_id >= thread_num_) {
    MS_LOG(ERROR) << "task_id " << task_id << " is out of range, node is " << name_;
    return RET_ERROR;
  }
  int start_oc = split_points_[task_id];
  int end_oc = col_step_;
  if (task_id < (thread_num_ - 1)) {
    end_oc = split_points_[task_id + 1];
  }
  int compute_oc = end_oc - start_oc;
  if (compute_oc <= 0) {
    return RET_OK;
  }
  int func_flag{0};
  if (params_->row_ == 1) {
    func_flag += (!params_->b_const_ && params_->col_ <= C128NUM) ? C2NUM : C1NUM;
  }
  int b_stride = func_flag == C2NUM ? 1 : params_->deep_;
  for (int i = 0; i < params_->batch; ++i) {
    auto a = matrix_a_.pack_ptr + a_offset_[i] * params_->row_align_ * params_->deep_;
    auto b = matrix_b_.pack_ptr + b_offset_[i] * params_->deep_ * params_->col_align_ + start_oc * b_stride;
    auto c = output_data_ + i * params_->row_ * col_step_ + start_oc;
    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr + start_oc;
    if (func_flag == 0) {
      if (out_need_aligned_) {
        MatMulAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, params_->col_align_,
                         params_->row_);
      } else {
        MatMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, params_->col_,
                             params_->row_);
      }
    } else if (func_flag == C1NUM) {
      if (out_need_aligned_) {
        MatVecMulAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, params_->col_align_);
      } else {
        MatVecMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, params_->col_);
      }
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, col_step_);
    }
  }
  return RET_OK;
}

// vec * vec
int MatmulFp32AVX512CPUKernel::ParallelRunByGEPDOT(int task_id) const {
  if (task_id < 0 || task_id >= thread_num_) {
    MS_LOG(ERROR) << "task_id " << task_id << " is out of range, node is " << name_;
    return RET_ERROR;
  }

  // by BatchCut
  int start_batch = task_id * batch_stride_;
  int end_batch = start_batch + batch_stride_;
  float bias = 0;
  if (matrix_c_.pack_ptr != nullptr) {
    bias = matrix_c_.pack_ptr[0];
  }
  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * params_->row_ * params_->deep_;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * params_->deep_ * params_->col_;
    float *c = output_data_ + index * params_->row_ * params_->col_;
    gemmIsNotPackFun(a, b, c, &bias, params_->row_, params_->deep_, params_->act_type_);
  }

  // by RowCut
  int split_points_size = static_cast<int>(row_split_points_.size()) - 1;
  if (task_id >= split_points_size) {
    return RET_OK;
  }
  for (int index = thread_num_ * batch_stride_; index < params_->batch; ++index) {
    int start_row = row_split_points_[task_id];
    int end_row = row_split_points_[task_id + 1];
    int row_num = end_row - start_row;
    if (row_num <= 0) {
      continue;
    }
    const float *a =
      matrix_a_.pack_ptr + a_offset_[index] * params_->row_ * params_->deep_ + start_row * params_->deep_;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * params_->deep_ * params_->col_;
    float *c = output_data_ + index * params_->row_ * params_->col_ + start_row * col_step_;

    gemmIsNotPackFun(a, b, c, &bias, row_num, params_->deep_, params_->act_type_);
  }

  return RET_OK;
}

// vec * vec
int MatmulFp32AVX512CPUKernel::ParallelRunByRow1Deep1GEPDOT(int task_id) const {
  auto a_plane_size = params_->row_align_ * params_->deep_;
  auto b_plane_size = params_->deep_ * params_->col_align_;
  auto c_plane_size = params_->row_ * col_step_;
  int matrix_col = col_step_;
  int matrix_deep = params_->deep_;

  // by BatchCut
  int start_batch = task_id * batch_stride_;
  int end_batch = MSMIN(params_->batch, start_batch + batch_stride_);

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * a_plane_size;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * b_plane_size;
    float *c = output_data_ + index * c_plane_size;

    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr;
    gemmIsNotPackFun(a, b, c, bias, matrix_col, matrix_deep, params_->act_type_);
  }

  // by ColCut
  int col_split_points_size = static_cast<int>(col_split_points_.size());
  if (task_id < col_split_points_size) {
    int start_oc = col_split_points_[task_id];
    int end_oc = matrix_col;
    if (task_id < (col_split_points_size - 1)) {
      end_oc = col_split_points_[task_id + 1];
    }
    int compute_oc = end_oc - start_oc;
    if (compute_oc <= 0) {
      return RET_OK;
    }

    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr + start_oc;
    for (int i = thread_num_ * batch_stride_; i < params_->batch; ++i) {
      auto a = matrix_a_.pack_ptr + a_offset_[i] * a_plane_size;
      auto b = matrix_b_.pack_ptr + b_offset_[i] * b_plane_size + start_oc;
      auto c = output_data_ + i * c_plane_size + start_oc;

      gemmIsNotPackFun(a, b, c, bias, compute_oc, matrix_deep, params_->act_type_);
    }
  }
  return RET_OK;
}

// vec * mat
int MatmulFp32AVX512CPUKernel::ParallelRunByGEPM(int task_id) const {
  auto a_plane_size = params_->row_align_ * params_->deep_;
  auto b_plane_size = params_->deep_ * params_->col_align_;
  auto c_plane_size = params_->row_ * col_step_;
  int matrix_col = col_step_;
  int matrix_deep = params_->deep_;

  // by BatchCut
  int start_batch = task_id * batch_stride_;
  int end_batch = MSMIN(params_->batch, start_batch + batch_stride_);

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * a_plane_size;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * b_plane_size;
    float *c = output_data_ + index * c_plane_size;

    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr;
    MatVecMulNoPackFp32(a, b, c, bias, params_->act_type_, matrix_deep, matrix_col, matrix_col);
  }

  // by ColCut
  int col_split_points_size = static_cast<int>(col_split_points_.size());
  if (task_id < col_split_points_size) {
    int start_oc = col_split_points_[task_id];
    int end_oc = matrix_col;
    if (task_id < (col_split_points_size - 1)) {
      end_oc = col_split_points_[task_id + 1];
    }
    int compute_oc = end_oc - start_oc;
    if (compute_oc <= 0) {
      return RET_OK;
    }

    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr + start_oc;
    for (int i = thread_num_ * batch_stride_; i < params_->batch; ++i) {
      auto a = matrix_a_.pack_ptr + a_offset_[i] * a_plane_size;
      auto b = matrix_b_.pack_ptr + b_offset_[i] * b_plane_size + start_oc;
      auto c = output_data_ + i * c_plane_size + start_oc;

      MatVecMulNoPackFp32(a, b, c, bias, params_->act_type_, matrix_deep, compute_oc, matrix_col);
    }
  }
  return RET_OK;
}

// mat * mat
int MatmulFp32AVX512CPUKernel::ParallelRunByGEMM(int task_id) const {
  if (task_id < 0 || task_id >= thread_num_) {
    MS_LOG(ERROR) << "task_id " << task_id << " is out of range, node is " << name_;
    return RET_ERROR;
  }

  auto a_plane_size = params_->row_align_ * params_->deep_;
  auto b_plane_size = params_->deep_ * params_->col_align_;
  auto c_plane_size = params_->row_ * col_step_;
  int matrix_row = params_->row_;
  int matrix_col = col_step_;
  int matrix_deep = params_->deep_;

  // by BatchCut
  int start_batch = task_id * batch_stride_;
  int end_batch = start_batch + batch_stride_;
  auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr;
  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * a_plane_size;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * b_plane_size;
    float *c = output_data_ + index * c_plane_size;

    MatMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, matrix_deep, matrix_col, matrix_col, matrix_row);
  }

  // by ColCut
  int col_split_points_size = static_cast<int>(col_split_points_.size()) - 1;
  if (task_id < col_split_points_size) {
    int start_oc = col_split_points_[task_id];
    int end_oc = col_split_points_[task_id + 1];
    int compute_oc = end_oc - start_oc;

    bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr + start_oc;
    if (compute_oc > 0) {
      for (int i = thread_num_ * batch_stride_; i < params_->batch; ++i) {
        auto a = matrix_a_.pack_ptr + a_offset_[i] * a_plane_size;
        auto b = matrix_b_.pack_ptr + b_offset_[i] * b_plane_size + start_oc * matrix_deep;
        auto c = output_data_ + i * c_plane_size + start_oc;

        MatMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, matrix_deep, compute_oc, matrix_col, matrix_row);
      }
    }
  }

  // by RowCut
  int start_oc = col_split_points_[col_split_points_size];
  int end_oc = matrix_col;
  int compute_oc = end_oc - start_oc;
  if (compute_oc <= 0) {
    return RET_OK;
  }

  int row_split_points_size = static_cast<int>(row_split_points_.size()) - 1;
  if (task_id >= row_split_points_size) {
    return RET_OK;
  }
  int start_row = row_split_points_[task_id];
  int end_row = row_split_points_[task_id + 1];
  int row_num = end_row - start_row;
  if (row_num <= 0) {
    return RET_OK;
  }

  bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr + start_oc;
  for (int i = thread_num_ * batch_stride_; i < params_->batch; ++i) {
    auto a = matrix_a_.pack_ptr + a_offset_[i] * a_plane_size + start_row * matrix_deep;
    auto b = matrix_b_.pack_ptr + b_offset_[i] * b_plane_size + start_oc * matrix_deep;
    auto c = output_data_ + i * c_plane_size + start_row * matrix_col + start_oc;

    MatMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, matrix_deep, compute_oc, matrix_col, row_num);
  }

  return RET_OK;
}

// mat * mat
int MatmulFp32AVX512CPUKernel::ParallelRunByBatchColRowGEMM(int task_id) const {
  if (task_id < 0 || task_id >= thread_num_) {
    MS_LOG(ERROR) << "task_id " << task_id << " is out of range, node is " << name_;
    return RET_ERROR;
  }

  auto a_plane_size = params_->row_align_ * params_->deep_;
  auto b_plane_size = params_->deep_ * params_->col_align_;
  auto c_plane_size = params_->row_ * col_step_;
  int matrix_row = params_->row_;
  int matrix_col = col_step_;
  int matrix_deep = params_->deep_;

  // by BatchCut
  int start_batch = task_id * batch_stride_;
  int end_batch = start_batch + batch_stride_;
  auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr;
  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * a_plane_size;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * b_plane_size;
    float *c = output_data_ + index * c_plane_size;

    MatMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, matrix_deep, matrix_col, matrix_col, matrix_row);
  }

  for (auto matmul_slice : matmul_slice_set_.at(task_id)) {
    int start_oc = matmul_slice.col_s_;
    int end_oc = matmul_slice.col_e_;
    int compute_oc = end_oc - start_oc;
    if (compute_oc <= 0) {
      return RET_OK;
    }

    int start_row = matmul_slice.row_s_;
    int end_row = matmul_slice.row_e_;
    int row_num = end_row - start_row;
    if (row_num <= 0) {
      return RET_OK;
    }

    bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr + start_oc;
    for (int i = thread_num_ * batch_stride_; i < params_->batch; ++i) {
      auto a = matrix_a_.pack_ptr + a_offset_[i] * a_plane_size + start_row * matrix_deep;
      auto b = matrix_b_.pack_ptr + b_offset_[i] * b_plane_size + start_oc * matrix_deep;
      auto c = output_data_ + i * c_plane_size + start_row * matrix_col + start_oc;

      MatMulMaskAvx512Fp32(a, b, c, bias, params_->act_type_, matrix_deep, compute_oc, matrix_col, row_num);
    }
  }

  return RET_OK;
}

void MatmulFp32AVX512CPUKernel::BatchRowThreadCut() {
  // BatchCut
  batch_stride_ = DOWN_DIV(params_->batch, thread_num_);

  // RowCut
  int row_step = MSMAX(params_->row_ / thread_num_, row_min_unit_);
  int row_remaining = params_->row_ - row_step * thread_num_;
  row_split_points_.clear();
  int row_split_point = 0;
  while (row_split_point < params_->row_) {
    row_split_points_.push_back(row_split_point);
    row_split_point += row_step;
    if (row_remaining > 0) {
      ++row_split_point;
      --row_remaining;
    }
  }
  row_split_points_.push_back(params_->row_);
  if (batch_stride_ == 0) {
    thread_num_ = row_split_points_.size() - 1;
  }
}

void MatmulFp32AVX512CPUKernel::BatchColThreadCut() {
  // BatchCut
  batch_stride_ = DOWN_DIV(params_->batch, thread_num_);

  // ColCut
  int total_col_unit = UP_DIV(params_->col_align_, col_min_unit_);
  auto thread_num_tmp = MSMIN(thread_num_, total_col_unit);
  int block_col_unit = UP_DIV(total_col_unit, thread_num_tmp);
  col_split_points_.clear();
  int split_point = 0;
  while (split_point < total_col_unit) {
    col_split_points_.push_back(split_point * col_min_unit_);
    split_point += block_col_unit;
  }
  if (batch_stride_ == 0) {
    thread_num_ = col_split_points_.size();
  }
}

void MatmulFp32AVX512CPUKernel::BatchColRowThreadCut() {
  // BatchCut
  batch_stride_ = DOWN_DIV(params_->batch, thread_num_);

  // ColCut
  int total_col_unit = UP_DIV(params_->col_align_, col_min_unit_);
  block_col_unit_ = DOWN_DIV(total_col_unit, thread_num_);
  col_split_points_.clear();
  col_split_points_.push_back(0);
  if (block_col_unit_ > 0) {
    int col_split_point = 0;
    for (int i = 0; i < thread_num_; i++) {
      col_split_point += block_col_unit_;
      col_split_points_.push_back(MSMIN(col_split_point * col_min_unit_, col_step_));
    }
  }

  // RowCut
  int row_step = MSMAX(params_->row_ / thread_num_, row_min_unit_);
  int row_remaining = params_->row_ - row_step * thread_num_;
  row_split_points_.clear();
  int row_split_point = 0;
  while (row_split_point < params_->row_) {
    row_split_points_.push_back(row_split_point);
    row_split_point += row_step;
    if (row_remaining > 0) {
      ++row_split_point;
      --row_remaining;
    }
  }
  row_split_points_.push_back(params_->row_);
  if (batch_stride_ == 0 && block_col_unit_ == 0) {
    thread_num_ = row_split_points_.size() - 1;
  }
}

void MatmulFp32AVX512CPUKernel::BatchColRowSliceThreadCut() {
  // BatchCut
  batch_stride_ = DOWN_DIV(params_->batch, thread_num_);

  int row_s = 0;
  int row_e = params_->row_;
  int col_s = 0;
  int col_e = params_->col_;

  // ColCut
  int total_col_unit = UP_DIV(params_->col_align_, col_min_unit_);
  block_col_unit_ = DOWN_DIV(total_col_unit, thread_num_);
  col_split_points_.clear();
  col_split_points_.push_back(0);
  if (block_col_unit_ > 0) {
    int col_split_point = 0;
    for (int i = 0; i < thread_num_; i++) {
      MatmulSlice matmul_slice;
      matmul_slice.row_s_ = row_s;
      matmul_slice.row_e_ = row_e;
      matmul_slice.col_s_ = col_split_point * col_min_unit_;
      col_split_point += block_col_unit_;
      col_s = MSMIN(col_split_point * col_min_unit_, col_step_);
      matmul_slice.col_e_ = col_s;
      matmul_slice_set_[i].push_back(matmul_slice);
    }
  }
  if (col_e - col_s <= 0) {
    return;
  }

  // RowColCut
  int row_thread = 0;

  auto less_col_align = UP_ROUND(col_e - col_s, C16NUM);
  bool use_colrowcut_flag = ((less_col_align / C64NUM) * C64NUM) == less_col_align;
  bool use_rowcut_flag = params_->row_ >= C6NUM * thread_num_ || col_e - col_s <= C64NUM;
  if (use_rowcut_flag && !use_colrowcut_flag) {
    int row_step = MSMAX(params_->row_ / thread_num_, row_min_unit_);
    int row_remaining = params_->row_ - row_step * thread_num_;
    int row_split_point = 0;

    for (row_thread = 0; row_thread < thread_num_ && row_split_point < params_->row_; row_thread++) {
      MatmulSlice matmul_slice;
      matmul_slice.row_s_ = row_split_point;

      row_split_point += row_step;
      if (row_remaining > 0) {
        ++row_split_point;
        --row_remaining;
      }

      matmul_slice.row_e_ = row_split_point;
      matmul_slice.col_s_ = col_s;
      matmul_slice.col_e_ = col_e;
      matmul_slice_set_[row_thread].push_back(matmul_slice);
    }
  } else {
    auto col_num = UP_DIV(col_e - col_s, C64NUM);
    auto row_num = MSMIN(UP_DIV(thread_num_, col_num), (row_e - row_s));
    auto tile_remaining = MSMAX(col_num * row_num - thread_num_, 0);

    int row_step = (row_e - row_s) / row_num;
    int row_remaining_tmp = (row_e - row_s) - row_step * row_num;

    int row_step_cut2 = (row_num == 1) ? row_step : (row_e - row_s) / (row_num - 1);
    int row_remaining_cut2_tmp = (row_e - row_s) - row_step_cut2 * (row_num - 1);

    MatmulSlice matmul_slice;
    for (int c = 0; c < col_num; c++) {
      matmul_slice.col_s_ = col_s + c * C64NUM;
      matmul_slice.col_e_ = MSMIN(col_s + (c + 1) * C64NUM, params_->col_);
      int row_split_point = 0;
      int row_remaining = row_remaining_tmp;
      int row_remaining_cut2 = row_remaining_cut2_tmp;
      if (c < col_num - tile_remaining) {
        for (int r = 0; r < row_num; r++) {
          matmul_slice.row_s_ = row_split_point;
          row_split_point += row_step;
          if (row_remaining > 0) {
            ++row_split_point;
            --row_remaining;
          }
          matmul_slice.row_e_ = MSMIN(row_split_point, params_->row_);
          matmul_slice_set_[row_thread++].push_back(matmul_slice);
        }
      } else {
        for (int r = 0; r < row_num - 1; r++) {
          matmul_slice.row_s_ = row_split_point;
          row_split_point += row_step_cut2;
          if (row_remaining_cut2 > 0) {
            ++row_split_point;
            --row_remaining_cut2;
          }
          matmul_slice.row_e_ = MSMIN(row_split_point, params_->row_);
          matmul_slice_set_[row_thread++].push_back(matmul_slice);
        }
      }
    }
  }
  if ((batch_stride_ == 0) && (block_col_unit_ == 0)) {
    thread_num_ = row_thread;
  }
}  // namespace mindspore::kernel

int MatmulFp32AVX512CPUKernel::GetThreadCuttingPolicy() {
  size_t total_cost = static_cast<size_t>(params_->batch) * static_cast<size_t>(params_->row_) *
                      static_cast<size_t>(params_->col_) * static_cast<size_t>(params_->deep_);

  // Thread Update
  thread_num_ = MSMAX(MSMIN(static_cast<int>(total_cost / min_calc_cost_), op_parameter_->thread_num_), C1NUM);

  if (params_->deep_ < C128NUM) {
    return MatmulFp32BaseCPUKernel::GetThreadCuttingPolicy();
  }
  matmul_slice_set_.clear();
  matmul_slice_set_.resize(thread_num_);

  if (params_->col_ == 1 && !params_->a_const_) {
    BatchRowThreadCut();
    if (params_->deep_ == 1) {
      gemmIsNotPackFun = GemmIsNotPack;
    } else {
      gemmIsNotPackFun = GemmIsNotPackOptimize;
    }
    MatmulFp32BaseCPUKernel::SetRunByGEPDOT();
  } else if (params_->row_ == 1 && !params_->b_const_) {
    if (params_->deep_ == 1) {
      BatchColThreadCut();
      MatmulFp32BaseCPUKernel::SetRunByRow1Deep1GEPDOT();
      if (matrix_c_.pack_ptr != nullptr) {
        gemmIsNotPackFun = Row1Deep1GemmIsNotPack;
      } else {
        gemmIsNotPackFun = Row1Deep1NoBiasGemmIsNotPack;
      }

      return RET_OK;
    }
    BatchColThreadCut();
    MatmulFp32BaseCPUKernel::SetRunByGEPM();
  } else {
    BatchColRowSliceThreadCut();
    MatmulFp32BaseCPUKernel::SetRunByBatchColRowGEMM();
  }
  return RET_OK;
}

bool MatmulFp32AVX512CPUKernel::CheckThreadCuttingByRow() {
  if (b_batch_ != C1NUM) {
    return false;
  }
  if (row_num_ < thread_num_) {
    return false;
  }
  if (params_->col_ == 1) {
    row_min_unit_ = C8NUM;
    return true;
  }
  if (params_->row_ == 1 && !params_->b_const_ && params_->col_ <= C128NUM) {
    return false;
  }
  row_min_unit_ = C6NUM;
  if (col_step_ < C48NUM) {
    row_min_unit_ = C12NUM;
  } else if (col_step_ < C64NUM) {
    row_min_unit_ = C8NUM;
  }
  return MSMIN(row_num_ / row_min_unit_, thread_num_) > MSMIN(col_step_ / col_min_unit_, thread_num_);
}
}  // namespace mindspore::kernel
#endif
