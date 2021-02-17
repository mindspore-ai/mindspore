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

#include "src/runtime/kernel/arm/fp32/batchnorm_fp32.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNorm;

namespace mindspore::kernel {
int BatchnormCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int BatchnormCPUKernel::ReSize() {
  FreeMeanAndVariance();
  FillParam();
  return InitConstTensor();
}

void BatchnormCPUKernel::FreeMeanAndVariance() {
  if (mean_ != nullptr) {
    free(mean_);
    mean_ = nullptr;
  }
  if (variance_ != nullptr) {
    free(variance_);
    variance_ = nullptr;
  }
}

void BatchnormCPUKernel::FillParam() {
  auto input_shapes = in_tensors_.at(0)->shape();
  auto n_dim = input_shapes.size();
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  param->channel_ = input_shapes[n_dim - 1];
  param->unit_ = 1;
  for (size_t i = 0; i < n_dim - 1; i++) {
    param->unit_ *= input_shapes[i];
  }
  if (default_momentum_ < 0.0f) {
    default_momentum_ = param->momentum_;
  }
}

int BatchnormCPUKernel::InitConstTensor() {
  mean_ = malloc(in_tensors_.at(1)->Size());
  variance_ = malloc(in_tensors_.at(2)->Size());
  if (mean_ == nullptr || variance_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    FreeMeanAndVariance();
    return RET_ERROR;
  }
  memcpy(mean_, in_tensors_.at(1)->MutableData(), in_tensors_.at(1)->Size());
  memcpy(variance_, in_tensors_.at(2)->MutableData(), in_tensors_.at(2)->Size());
  return RET_OK;
}

int BatchnormCPUKernel::Run() {
  auto ret = ParallelLaunch(this->context_->thread_pool_, BatchNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
  }
  return ret;
}

int BatchnormCPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  BatchNormFp32(in_tensors_.at(0)->MutableData(), mean_, variance_, param, task_id, out_tensors_.at(0)->MutableData());
  return RET_OK;
}

int BatchNormRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<BatchnormCPUKernel *>(cdata);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int BatchnormCPUKernel::set_momentum(float momentum) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  param->momentum_ = momentum;

  return RET_OK;
}

float BatchnormCPUKernel::get_momentum() {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  return param->momentum_;
}

int BatchnormCPUKernel::RestoreDefaultMomentum() {
  set_momentum(default_momentum_);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchNorm, LiteKernelCreator<BatchnormCPUKernel>)
}  // namespace mindspore::kernel
