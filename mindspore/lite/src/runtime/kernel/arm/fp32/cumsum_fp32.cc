/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/arm/fp32/cumsum_fp32.h"
#include "nnacl/fp32/cumsum_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_CumSum;

namespace mindspore::kernel {
namespace {
int CumsumLaunch(void *cdata, int task_id) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "cdata is nullptr!";
    return RET_NULL_PTR;
  }
  auto kernel = reinterpret_cast<CumSumCPUKernel *>(cdata);
  auto input_tensor = kernel->in_tensors().at(0);
  int ret;
  if (input_tensor->data_type() == kNumberTypeFloat32) {
    ret = kernel->DoCumsum(task_id);
  } else if (input_tensor->data_type() == kNumberTypeInt32) {
    ret = kernel->DoCumsumInt(task_id);
  } else {
    MS_LOG(ERROR) << "Cumsum support data type int32 or float32";
    return RET_ERROR;
  }
  return ret;
}
}  // namespace

int CumSumCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CumSumCPUKernel::ReSize() {
  MS_ASSERT(in_tensors_.size() == 2);
  auto input_tensor = in_tensors_.at(0);
  auto axis_tensor = in_tensors_.at(1);
  int *axis_data = reinterpret_cast<int *>(axis_tensor->data_c());
  if (axis_data == nullptr) {
    MS_LOG(ERROR) << "Cumsum axis nullptr";
    return RET_ERROR;
  }
  param_->axis_ = *axis_data;
  if (param_->axis_ < 0) {
    param_->axis_ += in_tensors_.at(0)->shape().size();
  }
  if (static_cast<int>(in_tensors_.at(0)->shape().size()) <= param_->axis_) {
    MS_LOG(ERROR) << "axis " << param_->axis_ << " larger than in tensor rank " << in_tensors_.at(0)->shape().size();
    return RET_ERROR;
  }
  out_dim_ = 1;
  for (int i = 0; i < param_->axis_; ++i) {
    out_dim_ *= input_tensor->shape().at(i);
  }
  axis_dim_ = input_tensor->shape().at(param_->axis_);
  in_dim_ = 1;
  for (int i = param_->axis_ + 1; i < static_cast<int>(input_tensor->shape().size()); ++i) {
    in_dim_ *= input_tensor->shape().at(i);
  }
  unit_ = UP_DIV(out_dim_, op_parameter_->thread_num_);
  return RET_OK;
}

int CumSumCPUKernel::DoCumsum(int task_id) {
  auto input_tensor = in_tensors_.at(0);
  MS_ASSERT(input_tensor != nullptr);
  float *input_data = reinterpret_cast<float *>(input_tensor->data_c());
  if (input_data == nullptr) {
    MS_LOG(ERROR) << "input data nullptr";
    return RET_ERROR;
  }
  auto output_tensor = out_tensors_.at(0);
  MS_ASSERT(output_tensor != nullptr);
  float *output_data = reinterpret_cast<float *>(output_tensor->data_c());
  if (output_data == nullptr) {
    MS_LOG(ERROR) << "output data nullptr";
    return RET_ERROR;
  }
  float *input = input_data + task_id * unit_ * axis_dim_ * in_dim_;
  int out_dim = MSMIN(out_dim_ - unit_ * task_id, unit_);
  float *output = output_data + task_id * unit_ * axis_dim_ * in_dim_;
  if (!param_->reverse_) {
    Cumsum(input, output, out_dim, axis_dim_, in_dim_, param_->exclusive_);
  } else {
    CumsumReverse(input, output, out_dim, axis_dim_, in_dim_, param_->exclusive_);
  }
  return RET_OK;
}

int CumSumCPUKernel::DoCumsumInt(int task_id) {
  auto input_tensor = in_tensors_.at(0);
  MS_ASSERT(input_tensor != nullptr);
  int *input_data = reinterpret_cast<int *>(input_tensor->data_c());
  if (input_data == nullptr) {
    MS_LOG(ERROR) << "input data nullptr";
    return RET_ERROR;
  }
  auto output_tensor = out_tensors_.at(0);
  MS_ASSERT(output_tensor != nullptr);
  int *output_data = reinterpret_cast<int *>(output_tensor->data_c());
  if (output_data == nullptr) {
    MS_LOG(ERROR) << "output data nullptr";
    return RET_ERROR;
  }
  int *input = input_data + task_id * unit_ * axis_dim_ * in_dim_;
  int out_dim = MSMIN(out_dim_ - unit_ * task_id, unit_);
  int *output = output_data + task_id * unit_ * axis_dim_ * in_dim_;
  if (!param_->reverse_) {
    CumsumInt(input, output, out_dim, axis_dim_, in_dim_, param_->exclusive_);
  } else {
    CumsumReverseInt(input, output, out_dim, axis_dim_, in_dim_, param_->exclusive_);
  }
  return RET_OK;
}

int CumSumCPUKernel::Run() {
  int ret = ParallelLaunch(static_cast<const lite::InnerContext *>(this->context_)->thread_pool_, CumsumLaunch, this,
                           op_parameter_->thread_num_);

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Crop launch fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_CumSum, LiteKernelCreator<CumSumCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_CumSum, LiteKernelCreator<CumSumCPUKernel>)
}  // namespace mindspore::kernel
