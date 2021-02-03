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

#include "src/runtime/kernel/arm/fp32/fused_batchnorm_fp32.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FusedBatchNorm;

namespace mindspore::kernel {
int FusedBatchnormCPUKernel::ReSize() {
  FreeMeanAndVariance();
  FreeScaleAndOffset();
  FillParam();
  return InitConstTensor();
}

void FusedBatchnormCPUKernel::FreeScaleAndOffset() {
  if (scale_ != nullptr) {
    free(scale_);
    scale_ = nullptr;
  }
  if (offset_ != nullptr) {
    free(offset_);
    offset_ = nullptr;
  }
}

int FusedBatchnormCPUKernel::InitConstTensor() {
  auto scale = in_tensors_.at(1);
  auto offset = in_tensors_.at(2);
  auto mean = in_tensors_.at(3);
  auto variance = in_tensors_.at(4);

  scale_ = malloc(scale->Size());
  offset_ = malloc(offset->Size());
  mean_ = malloc(mean->Size());
  variance_ = malloc(variance->Size());

  if (scale_ == nullptr || offset_ == nullptr || mean_ == nullptr || variance_ == nullptr) {
    FreeMeanAndVariance();
    FreeScaleAndOffset();
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  memcpy(scale_, scale->MutableData(), scale->Size());
  memcpy(offset_, offset->MutableData(), offset->Size());
  memcpy(mean_, mean->MutableData(), mean->Size());
  memcpy(variance_, variance->MutableData(), variance->Size());

  return RET_OK;
}

int FusedBatchnormCPUKernel::Run() {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  if (IsTrain() && is_trainable() && in_tensors_.size() >= 5) {
    float *in = static_cast<float *>(in_tensors_[0]->MutableData());
    float *scale = static_cast<float *>(in_tensors_[1]->MutableData());
    float *offset = static_cast<float *>(in_tensors_[2]->MutableData());
    float *current_mean = static_cast<float *>(mean_);
    float *current_var = static_cast<float *>(variance_);
    float *save_mean = static_cast<float *>(in_tensors_[3]->MutableData());
    float *save_variance = static_cast<float *>(in_tensors_[4]->MutableData());

    std::fill(current_mean, current_mean + in_tensors_[3]->ElementsNum(), 0.f);
    std::fill(current_var, current_var + in_tensors_[4]->ElementsNum(), 0.f);
    FusedBatchNormFp32MeanVar(in, current_mean, current_var, param, static_cast<float *>(save_mean),
                              static_cast<float *>(save_variance));

    memcpy(out_tensors_.at(1)->MutableData(), scale, out_tensors_.at(1)->Size());
    memcpy(out_tensors_.at(2)->MutableData(), offset, out_tensors_.at(2)->Size());
    memcpy(out_tensors_.at(3)->MutableData(), current_mean, out_tensors_.at(3)->Size());
    memcpy(out_tensors_.at(4)->MutableData(), current_var, out_tensors_.at(4)->Size());

    // Copy to local variables
    memcpy(scale_, scale, in_tensors_[1]->Size());
    memcpy(offset_, offset, in_tensors_[2]->Size());

    trained_ = true;  // trained at least once
  }
  auto ret = ParallelLaunch(this->context_->thread_pool_, BatchNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
  }

  return ret;
}

int FusedBatchnormCPUKernel::Eval() {
  LiteKernel::Eval();
  if (trained_) {
    float *save_mean = static_cast<float *>(in_tensors_.at(3)->MutableData());
    float *save_var = static_cast<float *>(in_tensors_.at(4)->MutableData());
    float *scale = static_cast<float *>(in_tensors_.at(1)->MutableData());
    float *bias = static_cast<float *>(in_tensors_.at(2)->MutableData());

    // Copy to local variables
    memcpy(scale_, scale, in_tensors_.at(1)->Size());
    memcpy(offset_, bias, in_tensors_.at(2)->Size());
    memcpy(mean_, save_mean, in_tensors_.at(3)->Size());
    memcpy(variance_, save_var, in_tensors_.at(4)->Size());
  }
  return RET_OK;
}

int FusedBatchnormCPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  FusedBatchNormFp32(in_tensors_.at(0)->MutableData(), scale_, offset_, mean_, variance_, param, task_id,
                     out_tensors_.at(0)->MutableData());
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FusedBatchNorm, LiteKernelCreator<FusedBatchnormCPUKernel>)
}  // namespace mindspore::kernel
