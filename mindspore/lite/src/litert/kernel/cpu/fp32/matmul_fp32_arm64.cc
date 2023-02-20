#ifdef ENABLE_ARM64
/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/matmul_fp32_arm64.h"
#include <vector>
#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/pack_fp32_opt.h"

namespace mindspore::kernel {
namespace {
constexpr int64_t kPackAMinUnitNum = 1 << 13;
}  // namespace
void MatmulFp32ARM64CPUKernel::InitGlobalVariable() {
  matrix_a_.need_pack = true;
  matrix_b_.need_pack = true;
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2Row12MajorParallel : RowMajor2Col12MajorParallel;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col8MajorParallel : RowMajor2Row8MajorParallel;
  pack_opt_ = true;
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
  col_min_unit_ = C8NUM;
}

int MatmulFp32ARM64CPUKernel::PackMatrixAImplOpt() {
  auto src_ptr =
    matrix_a_.has_origin ? matrix_a_.origin_ptr : reinterpret_cast<float *>(in_tensors_[FIRST_INPUT]->data());
  MS_CHECK_TRUE_MSG(src_ptr != nullptr, RET_ERROR, "matrix-a source ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_a_.pack_ptr != nullptr, RET_ERROR, "matrix-a pack ptr is a nullptr.");
  int64_t unit_num{0};
  unit_num = a_batch_ * UP_DIV(params_->row_, C12NUM) * params_->deep_;
  int thread_count = MSMIN(op_parameter_->thread_num_, UP_DIV(unit_num, kPackAMinUnitNum));
  if (thread_count < 1) {
    thread_count = 1;
  }
  int64_t block_size = unit_num / thread_count;
  int64_t remain_size = unit_num - block_size * thread_count;
  std::vector<int64_t> points;
  int64_t start = 0;
  while (start < unit_num) {
    points.push_back(start);
    start += block_size;
    if (remain_size > 0) {
      ++start;
      --remain_size;
    }
  }
  thread_count = points.size();
  auto Pack = [&points, unit_num, src_ptr, this](void *, int task_id, float, float) {
    int64_t start = points[task_id];
    int64_t end = unit_num;
    if (task_id < static_cast<int>(points.size()) - 1) {
      end = points[task_id + 1];
    }
    if (params_->a_transpose_) {
      RowMajor2Row12MajorOpt(src_ptr, matrix_a_.pack_ptr, params_->deep_, params_->row_, start, end);
    } else {
      RowMajor2Col12MajorOpt(src_ptr, matrix_a_.pack_ptr, params_->row_, params_->deep_, start, end);
    }
    return RET_OK;
  };
  if (thread_count == 1) {
    return Pack(nullptr, 0, 0, 1);
  }
  return ParallelLaunch(this->ms_context_, Pack, nullptr, thread_count);
}

int MatmulFp32ARM64CPUKernel::ParallelRunByBatch(int task_id) const {
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
      MatMulOpt(a, b, c, bias, params_->act_type_, params_->deep_, params_->row_, col_step_, params_->col_,
                OutType_Nhwc);
    } else if (func_flag == C1NUM) {
      MatVecMulPackFp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, col_step_);
    }
  }
  return RET_OK;
}

int MatmulFp32ARM64CPUKernel::ParallelRunByRow(int task_id) const {
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
  GemmIsNotPackByRow(matrix_a_.pack_ptr, matrix_b_.pack_ptr, output_data_, matrix_c_.pack_ptr, start_row, end_row,
                     params_->deep_, params_->act_type_);
  return RET_OK;
}

int MatmulFp32ARM64CPUKernel::ParallelRunByOC(int task_id) const {
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
      MatMulOpt(a, b, c, bias, params_->act_type_, params_->deep_, params_->row_, compute_oc, params_->col_,
                OutType_Nhwc);
    } else if (func_flag == C1NUM) {
      MatVecMulPackFp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, col_step_);
    }
  }
  return RET_OK;
}

bool MatmulFp32ARM64CPUKernel::CheckThreadCuttingByRow() {
  if (b_batch_ != C1NUM) {
    return false;
  }
  if (params_->batch >= op_parameter_->thread_num_ || params_->col_ == 1) {
    row_min_unit_ = C4NUM;
    return true;
  }
  return false;
}
}  // namespace mindspore::kernel
#endif
