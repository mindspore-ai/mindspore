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

#include "src/runtime/kernel/arm/fp32/bias_fp32.h"
#include <vector>
#include "nnacl/fp32/bias_add.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BiasAdd;

namespace mindspore::kernel {
int BiasAddRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto kernel = reinterpret_cast<BiasCPUKernel *>(cdata);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int BiasCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int BiasCPUKernel::ReSize() {
  auto in_dims = in_tensors_.at(0)->shape();
  auto bias_dims = in_tensors_.at(1)->shape();
  if (bias_dims.empty() || in_dims.empty() || in_dims.size() < bias_dims.size()) {
    MS_LOG(ERROR) << "inTensors' shape are invalid.";
    return RET_ERROR;
  }
  size_t dim_offset = in_dims.size() - bias_dims.size();
  inner_num_ = 1;
  for (size_t i = 0; i < bias_dims.size(); ++i) {
    if (in_dims[i + dim_offset] != bias_dims[i]) {
      MS_LOG(ERROR) << "inTensors' shape cannot match.";
      return RET_ERROR;
    }
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(bias_dims[i], inner_num_), RET_ERROR, "mul overflow.");
    inner_num_ *= bias_dims[i];
  }
  outer_num_ = 1;
  for (size_t i = 0; i < dim_offset; ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(in_dims[i], outer_num_), RET_ERROR, "mul overflow.");
    outer_num_ *= in_dims[i];
  }
  MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(inner_num_, outer_num_), RET_ERROR, "mul overflow.");
  total_num_ = inner_num_ * outer_num_;
  GetThreadSegmentInfos();
  return RET_OK;
}

void BiasCPUKernel::GetThreadSegmentInfos() {
  split_start_points_ = std::vector<int64_t>(op_parameter_->thread_num_, 0);
  split_end_points_ = std::vector<int64_t>(op_parameter_->thread_num_, 0);
  int64_t step = MSMAX(total_num_ / op_parameter_->thread_num_, C128NUM);
  int64_t remain_data = MSMAX(total_num_ - step * op_parameter_->thread_num_, 0);
  for (int i = 0; i < op_parameter_->thread_num_; ++i) {
    if (i == 0) {
      split_end_points_[i] = MSMIN(step, total_num_) + (i < remain_data ? 1 : 0);
      continue;
    }
    split_start_points_[i] = split_end_points_[i - 1];
    if (split_start_points_[i] >= total_num_) {
      split_start_points_[i] = 0;
      break;
    }
    split_end_points_[i] =
      split_start_points_[i] + MSMIN(step, total_num_ - split_start_points_[i]) + (i < remain_data ? 1 : 0);
  }
  MS_ASSERT(inner_num_ != 0);
  if (inner_num_ >= C64NUM && step / inner_num_ >= C6NUM) {
    batch_priority_ = true;
  } else {
    batch_priority_ = false;
  }
}

int BiasCPUKernel::Run() {
  auto ret = ParallelLaunch(this->ms_context_, BiasAddRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BiasAddRun error error_code[" << ret << "]";
  }
  return ret;
}

int BiasCPUKernel::DoExecute(int task_id) {
  auto input = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto bias = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto output = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  if (split_start_points_[task_id] == split_end_points_[task_id]) {
    return lite::RET_OK;
  }
  BiasAddOpt(input, bias, output, split_start_points_[task_id], split_end_points_[task_id], inner_num_,
             batch_priority_);
  return lite::RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BiasAdd, LiteKernelCreator<BiasCPUKernel>)
}  // namespace mindspore::kernel
