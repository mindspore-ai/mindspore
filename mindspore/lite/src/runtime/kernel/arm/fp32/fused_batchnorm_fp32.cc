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
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_5D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
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
  auto scale = in_tensors_.at(SECOND_INPUT);
  auto offset = in_tensors_.at(THIRD_INPUT);
  auto mean = in_tensors_.at(FOURTH_INPUT);
  auto variance = in_tensors_.at(FIFTH_INPUT);

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

  CHECK_NULL_RETURN(scale->data());
  CHECK_NULL_RETURN(offset->data());
  CHECK_NULL_RETURN(mean->data());
  CHECK_NULL_RETURN(variance->data());
  memcpy(scale_, scale->data(), scale->Size());
  memcpy(offset_, offset->data(), offset->Size());
  memcpy(mean_, mean->data(), mean->Size());
  memcpy(variance_, variance->data(), variance->Size());
  return RET_OK;
}

int FusedBatchnormCPUKernel::Run() {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  MS_ASSERT(param != nullptr);
  if (IsTrain() && IsTrainable() && in_tensors_.size() >= DIMENSION_5D) {
    float *in = static_cast<float *>(in_tensors_.at(FIRST_INPUT)->data());
    float *scale = static_cast<float *>(in_tensors_.at(SECOND_INPUT)->data());
    float *offset = static_cast<float *>(in_tensors_.at(THIRD_INPUT)->data());
    float *current_mean = static_cast<float *>(mean_);
    float *current_var = static_cast<float *>(variance_);
    float *save_mean = static_cast<float *>(in_tensors_.at(FOURTH_INPUT)->data());
    float *save_variance = static_cast<float *>(in_tensors_.at(FIFTH_INPUT)->data());
    if (in == nullptr || scale == nullptr || offset == nullptr || current_mean == nullptr || current_var == nullptr ||
        save_mean == nullptr || save_variance == nullptr) {
      MS_LOG(ERROR) << "The input data is nullptr.";
      return RET_ERROR;
    }
    std::fill(current_mean, current_mean + in_tensors_.at(FOURTH_INPUT)->ElementsNum(), 0.f);
    std::fill(current_var, current_var + in_tensors_.at(FIFTH_INPUT)->ElementsNum(), 0.f);
    FusedBatchNormFp32MeanVar(in, current_mean, current_var, param, static_cast<float *>(save_mean),
                              static_cast<float *>(save_variance));

    CHECK_NULL_RETURN(out_tensors_.at(SECOND_INPUT)->data());
    CHECK_NULL_RETURN(out_tensors_.at(THIRD_INPUT)->data());
    CHECK_NULL_RETURN(out_tensors_.at(FOURTH_INPUT)->data());
    CHECK_NULL_RETURN(out_tensors_.at(FIFTH_INPUT)->data());
    memcpy(out_tensors_.at(SECOND_INPUT)->data(), scale, out_tensors_.at(SECOND_INPUT)->Size());
    memcpy(out_tensors_.at(THIRD_INPUT)->data(), offset, out_tensors_.at(THIRD_INPUT)->Size());
    memcpy(out_tensors_.at(FOURTH_INPUT)->data(), current_mean, out_tensors_.at(FOURTH_INPUT)->Size());
    memcpy(out_tensors_.at(FIFTH_INPUT)->data(), current_var, out_tensors_.at(FIFTH_INPUT)->Size());

    // Copy to local variables
    memcpy(scale_, scale, in_tensors_.at(SECOND_INPUT)->Size());
    memcpy(offset_, offset, in_tensors_.at(THIRD_INPUT)->Size());

    trained_ = true;  // trained at least once
  }
  auto ret = ParallelLaunch(this->ms_context_, BatchNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
  }
  return ret;
}

int FusedBatchnormCPUKernel::Eval() {
  InnerKernel::Eval();
  if (trained_) {
    float *save_mean = static_cast<float *>(in_tensors_.at(FOURTH_INPUT)->data());
    float *save_var = static_cast<float *>(in_tensors_.at(FIFTH_INPUT)->data());
    float *scale = static_cast<float *>(in_tensors_.at(SECOND_INPUT)->data());
    float *bias = static_cast<float *>(in_tensors_.at(THIRD_INPUT)->data());
    CHECK_NULL_RETURN(save_mean);
    CHECK_NULL_RETURN(save_var);
    CHECK_NULL_RETURN(scale);
    CHECK_NULL_RETURN(bias);

    // Copy to local variables
    memcpy(scale_, scale, in_tensors_.at(SECOND_INPUT)->Size());
    memcpy(offset_, bias, in_tensors_.at(THIRD_INPUT)->Size());
    memcpy(mean_, save_mean, in_tensors_.at(FOURTH_INPUT)->Size());
    memcpy(variance_, save_var, in_tensors_.at(FIFTH_INPUT)->Size());
  }
  return RET_OK;
}

int FusedBatchnormCPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  auto in_data = in_tensors_.at(FIRST_INPUT)->data();
  auto out_data = out_tensors_.at(FIRST_INPUT)->data();
  CHECK_NULL_RETURN(in_data);
  CHECK_NULL_RETURN(out_data);
  FusedBatchNormFp32(in_data, scale_, offset_, mean_, variance_, param, task_id, out_data);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FusedBatchNorm, LiteKernelCreator<FusedBatchnormCPUKernel>)
}  // namespace mindspore::kernel
