#ifdef ENABLE_ARM32
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

#include "src/runtime/kernel/arm/fp32/matmul_fp32_arm32.h"
#include "src/runtime/kernel/arm/fp32/matmul_fp32_base.h"
#include "nnacl/fp32/matmul_fp32.h"

namespace mindspore::kernel {
void MatmulFp32BaseCPUKernel::InitGlobalVariable() {
  matrix_a_.need_pack = true;
  matrix_b_.need_pack = true;
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2Row12Major : RowMajor2Col12Major;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col4Major : RowMajor2Row4Major;
  row_tile_ = C12NUM;
  col_tile_ = C4NUM;
  col_min_unit_ = C4NUM;
}

int MatmulFp32BaseCPUKernel::PackMatrixAImplOpt() {
  MS_LOG(ERROR) << "Matmul: don't support optimized-packing, only support single-thread currently.";
  return RET_ERROR;
}

int MatmulFp32BaseCPUKernel::ParallelRunByBatch(int task_id) const {
  int start_batch = task_id * batch_stride_;
  int end_batch = MSMIN(params_->batch, start_batch + batch_stride_);

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * params_->row_align_ * params_->deep_;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * params_->deep_ * params_->col_align_;
    float *c = output_data_ + index * params_->row_ * col_step_;

    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr;
    if (params_->row_ == 1) {
      MatVecMulFp32Block4(a, b, c, bias, params_->act_type_, params_->deep_, col_step_);
    } else {
      MatMulOpt(a, b, c, bias, params_->act_type_, params_->deep_, params_->row_, col_step_, params_->col_,
                OutType_Nhwc);
    }
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::ParallelRunByRow(int task_id) const { return RET_ERROR; }

int MatmulFp32BaseCPUKernel::ParallelRunByOC(int task_id) const {
  int start_oc = split_points_[task_id];
  int end_oc = col_step_;
  if (task_id < (thread_count_ - 1)) {
    end_oc = split_points_[task_id + 1];
  }
  int compute_oc = end_oc - start_oc;
  if (compute_oc <= 0) {
    return RET_OK;
  }
  for (int i = 0; i < params_->batch; ++i) {
    auto a = matrix_a_.pack_ptr + a_offset_[i] * params_->row_align_ * params_->deep_;
    auto b = matrix_b_.pack_ptr + b_offset_[i] * params_->deep_ * params_->col_align_ + start_oc * params_->deep_;
    auto c = output_data_ + i * params_->row_ * col_step_ + start_oc;
    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr + start_oc;
    if (params_->row_ == 1) {
      MatVecMulFp32Block4(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc);
    } else {
      MatMulOpt(a, b, c, bias, params_->act_type_, params_->deep_, params_->row_, compute_oc, params_->col_,
                OutType_Nhwc);
    }
  }
  return RET_OK;
}

bool MatmulFp32BaseCPUKernel::CheckThreadCuttingByRow() { return false; }
}  // namespace mindspore::kernel
#endif
