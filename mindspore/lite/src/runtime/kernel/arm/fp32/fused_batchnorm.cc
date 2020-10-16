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

#include "src/runtime/kernel/arm/fp32/fused_batchnorm.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
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
  if (save_mean_ != nullptr) {
    free(save_mean_);
    save_mean_ = nullptr;
  }
  if (save_variance_ != nullptr) {
    free(save_variance_);
    save_variance_ = nullptr;
  }
}

int FusedBatchnormCPUKernel::InitConstTensor() {
  auto scale = in_tensors_[1];
  auto offset = in_tensors_[2];
  auto mean = in_tensors_[3];
  auto variance = in_tensors_[4];

  scale_ = malloc(scale->Size());
  offset_ = malloc(offset->Size());
  mean_ = malloc(mean->Size());
  variance_ = malloc(variance->Size());
  save_mean_ = malloc(mean->Size());
  save_variance_ = malloc(variance->Size());

  if (scale_ == nullptr || offset_ == nullptr || mean_ == nullptr || variance_ == nullptr || save_mean_ == nullptr ||
      save_variance_ == nullptr) {
    FreeMeanAndVariance();
    FreeScaleAndOffset();
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  memcpy(scale_, scale->MutableData(), scale->Size());
  memcpy(offset_, offset->MutableData(), offset->Size());
  memcpy(mean_, mean->MutableData(), mean->Size());
  memcpy(variance_, variance->MutableData(), variance->Size());
  memset(save_mean_, 0, mean->Size());
  memset(save_variance_, 0, variance->Size());
  if (out_tensors_.size() > 4) {
    for (size_t i = 1; i < out_tensors_.size(); i++) {
      auto *data = static_cast<float *>(out_tensors_[i]->MutableData());
      std::fill(data, data + out_tensors_[i]->ElementsNum(), 0.f);
    }
  }

  return RET_OK;
}

int FusedBatchnormCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail! Ret error code: " << ret;
    return ret;
  }
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  if (is_train() && in_tensors_.size() >= 5) {
    float *in = static_cast<float *>(in_tensors_[0]->MutableData());
    float *scale = static_cast<float *>(in_tensors_[1]->MutableData());
    float *bias = static_cast<float *>(in_tensors_[2]->MutableData());
    float *mean = static_cast<float *>(in_tensors_[3]->MutableData());
    float *var = static_cast<float *>(in_tensors_[4]->MutableData());
    std::fill(mean, mean + in_tensors_[3]->ElementsNum(), 0.f);
    std::fill(var, var + in_tensors_[4]->ElementsNum(), 0.f);
    FusedBatchNormFp32MeanVar(in, mean, var, param, static_cast<float *>(save_mean_),
                              static_cast<float *>(save_variance_));
    memcpy(out_tensors_[3]->MutableData(), save_mean_, out_tensors_[3]->Size());
    memcpy(out_tensors_[4]->MutableData(), save_variance_, out_tensors_[3]->Size());
    memcpy(mean_, mean, in_tensors_[3]->Size());
    memcpy(variance_, var, in_tensors_[4]->Size());
    memcpy(scale_, scale, in_tensors_[1]->Size());
    memcpy(offset_, bias, in_tensors_[2]->Size());
    trained_ = true;  // trained at least once
  }
  ret = ParallelLaunch(this->context_->thread_pool_, BatchNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
  }
  return ret;
}

void FusedBatchnormCPUKernel::eval() {
  LiteKernel::eval();
  if (trained_) {
    float *run_mean = static_cast<float *>(in_tensors_[3]->MutableData());
    float *run_var = static_cast<float *>(in_tensors_[4]->MutableData());
    float *scale = static_cast<float *>(in_tensors_[1]->MutableData());
    float *bias = static_cast<float *>(in_tensors_[2]->MutableData());
    // Copy to input tensors for Model export
    memcpy(run_mean, save_mean_, in_tensors_[3]->Size());
    memcpy(run_var, save_variance_, in_tensors_[4]->Size());
    // Copy to local variables
    memcpy(mean_, run_mean, in_tensors_[3]->Size());
    memcpy(variance_, run_var, in_tensors_[4]->Size());
    memcpy(scale_, scale, in_tensors_[1]->Size());
    memcpy(offset_, bias, in_tensors_[2]->Size());
  }
}

int FusedBatchnormCPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  FusedBatchNormFp32(in_tensors_.at(0)->MutableData(), scale_, offset_, mean_, variance_, param, task_id,
                     out_tensors_.at(0)->MutableData());
  return mindspore::lite::RET_OK;
}

kernel::LiteKernel *CpuFusedBatchnormKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs,
                                                   OpParameter *op_parameter, const lite::InnerContext *ctx,
                                                   const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  FusedBatchnormCPUKernel *kernel =
    new (std::nothrow) FusedBatchnormCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new FusedBatchnormCPUKernel fail!";
    free(op_parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FusedBatchNorm, CpuFusedBatchnormKernelCreator)
}  // namespace mindspore::kernel
