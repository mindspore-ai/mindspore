/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <vector>
#include <cmath>
#include "src/litert/kernel/cpu/fp32/l2_norm_fp32.h"
#include "include/errorcode.h"
#include "nnacl/fp32/l2_norm_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_L2NormalizeFusion;

namespace mindspore::kernel {
namespace {
const int kMaxThreadNum = 8;
}
int L2NormCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int L2NormCPUKernel::MallocTmpBuffer() {
  auto shape = in_tensors_.at(kInputIndex)->shape();
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, shape.size() * sizeof(int));
  l2_norm_param_->shape_ = reinterpret_cast<int *>(malloc(shape.size() * sizeof(int)));
  if (l2_norm_param_->shape_ == nullptr) {
    MS_LOG(ERROR) << "Malloc data failed";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, kMaxThreadNum * sizeof(float));
  tmp_sum_ = reinterpret_cast<float *>(malloc(kMaxThreadNum * sizeof(float)));
  if (tmp_sum_ == nullptr) {
    MS_LOG(ERROR) << "Malloc data failed";
    return RET_ERROR;
  }
  return RET_OK;
}

void L2NormCPUKernel::FreeTmpBuffer() {
  if (l2_norm_param_->shape_ != nullptr) {
    free(l2_norm_param_->shape_);
    l2_norm_param_->shape_ = nullptr;
  }
  if (tmp_sum_ != nullptr) {
    free(tmp_sum_);
    tmp_sum_ = nullptr;
  }
}

int L2NormCPUKernel::ReSize() {
  FreeTmpBuffer();
  auto ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  l2_norm_param_->data_num_ = in_tensors_.at(kInputIndex)->ElementsNum();
  CHECK_NOT_EQUAL_RETURN(out_tensors_.at(kOutputIndex)->ElementsNum(), l2_norm_param_->data_num_);
  auto shape = in_tensors_.at(kInputIndex)->shape();
  l2_norm_param_->shape_num_ = shape.size();
  for (size_t i = 0; i < shape.size(); i++) {
    l2_norm_param_->shape_[i] = shape[i];
  }
  for (size_t i = 0; i < l2_norm_param_->axis_num_; ++i) {
    if (l2_norm_param_->axis_[i] < 0) {
      l2_norm_param_->axis_[i] += static_cast<int>(shape.size());
    }
  }
  return RET_OK;
}

int L2NormCPUKernel::CalcSquareSum(int task_id) const {
  int unit = UP_DIV(l2_norm_param_->data_num_, op_parameter_->thread_num_);
  if (INT_MUL_OVERFLOW(task_id, unit)) {
    MS_LOG(ERROR) << "int mul overflow.";
    return RET_ERROR;
  }
  int begin = task_id * unit;
  int end = MSMIN(begin + unit, l2_norm_param_->data_num_);
  return CalcThreadSquareSum(input_ptr_, tmp_sum_ + task_id, begin, end);
}

int L2NormCPUKernel::DivSqrtSum(int task_id) const {
  int unit = UP_DIV(l2_norm_param_->data_num_, op_parameter_->thread_num_);
  if (INT_MUL_OVERFLOW(task_id, unit)) {
    MS_LOG(ERROR) << "int mul overflow.";
    return RET_ERROR;
  }
  int begin = task_id * unit;
  int end = MSMIN(begin + unit, l2_norm_param_->data_num_);
  return ThreadDivSqrtSum(input_ptr_, output_ptr_, l2_norm_param_, sqrt_sum_, begin, end);
}

int L2NormCPUKernel::CalcL2NormTrailingAxis(int task_id) const {
  auto input = in_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  MS_CHECK_TRUE_MSG(!input->shape().empty(), RET_ERROR, "Input shape of L2Norm is empty.");
  if (input->shape().back() == 0) {
    MS_LOG(ERROR) << "input->shape().back() is 0";
    return RET_ERROR;
  }
  int outer_size = input->ElementsNum() / input->shape().back();
  int unit = UP_DIV(outer_size, op_parameter_->thread_num_);
  if (INT_MUL_OVERFLOW(task_id, unit)) {
    MS_LOG(ERROR) << "int mul overflow.";
    return RET_ERROR;
  }
  int begin = task_id * unit;
  int end = MSMIN(begin + unit, outer_size);
  return ThreadTrailingAxis(input_ptr_, output_ptr_, l2_norm_param_, begin, end);
}

int SquareSumRun(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<const L2NormCPUKernel *>(cdata);
  auto ret = kernel->CalcSquareSum(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "L2Norm SquareSumRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int L2NormRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto kernel = reinterpret_cast<const L2NormCPUKernel *>(cdata);
  auto ret = kernel->DivSqrtSum(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "L2Norm L2NormRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int L2NormTrailingAxisRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto kernel = reinterpret_cast<const L2NormCPUKernel *>(cdata);
  auto ret = kernel->CalcL2NormTrailingAxis(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "L2Norm TrailingAxisRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int L2NormCPUKernel::Run() {
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  CHECK_NULL_RETURN(out_tensors_.at(kOutputIndex));
  auto input_shape = in_tensors().at(kInputIndex)->shape();
  input_ptr_ = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->MutableData());
  output_ptr_ = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  CHECK_NULL_RETURN(input_ptr_);
  CHECK_NULL_RETURN(output_ptr_);
  int ret;
  if (l2_norm_param_->axis_num_ == 0 || l2_norm_param_->axis_num_ == input_shape.size()) {
    // all axis
    ret = ParallelLaunch(this->ms_context_, SquareSumRun, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "L2Norm error: error_code[" << ret << "]";
      return RET_ERROR;
    }
    float sum = 0.0f;
    for (int i = 0; i < op_parameter_->thread_num_; ++i) {
      sum += tmp_sum_[i];
    }
    sqrt_sum_ = sqrt(sum > l2_norm_param_->epsilon_ ? sum : l2_norm_param_->epsilon_);
    ret = ParallelLaunch(this->ms_context_, L2NormRun, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "L2Norm error: error_code[" << ret << "]";
      return RET_ERROR;
    }
  } else if (l2_norm_param_->axis_num_ == 1 && l2_norm_param_->axis_[0] == static_cast<int>(input_shape.size()) - 1) {
    ret = ParallelLaunch(this->ms_context_, L2NormTrailingAxisRun, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "L2Norm error: error_code[" << ret << "]";
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "L2Norm only support reduce on all axis and trailing axis with trailing axis";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_L2NormalizeFusion, LiteKernelCreator<L2NormCPUKernel>)
}  // namespace mindspore::kernel
