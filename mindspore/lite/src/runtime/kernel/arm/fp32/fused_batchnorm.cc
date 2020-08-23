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

  if (scale_ == nullptr || offset_ == nullptr || mean_ == nullptr || variance_ == nullptr) {
    FreeMeanAndVariance();
    FreeScaleAndOffset();
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  memcpy(scale_, scale->Data(), scale->Size());
  memcpy(offset_, offset->Data(), offset->Size());
  memcpy(mean_, mean->Data(), mean->Size());
  memcpy(variance_, variance->Data(), variance->Size());
  return RET_OK;
}

int FusedBatchnormCPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  FusedBatchNormFp32(in_tensors_.at(0)->Data(), scale_, offset_, mean_, variance_, param, task_id,
                     out_tensors_.at(0)->Data());
  return mindspore::lite::RET_OK;
}

kernel::LiteKernel *CpuFusedBatchnormKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *op_parameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  FusedBatchnormCPUKernel *kernel =
    new (std::nothrow) FusedBatchnormCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new FusedBatchnormCPUKernel fail!";
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
