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
#include <cmath>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FusedBatchNorm;

namespace mindspore::kernel {
int FusedBatchnormCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    SetNeedReInit();
    return RET_OK;
  }
  input_shape_ = reinterpret_cast<int *>(malloc(sizeof(int) * inputs_[0]->shape().size()));
  memcpy(input_shape_, inputs_[0]->shape().data(), inputs_[0]->shape().size() * sizeof(int));
  return RET_OK;
}

int FusedBatchnormCPUKernel::ReSize() { return RET_OK; }

int FusedBatchnormCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto input_addr = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto scale_addr = reinterpret_cast<float *>(inputs_.at(1)->Data());
  auto offest_addr = reinterpret_cast<float *>(inputs_.at(2)->Data());
  auto mean_addr = reinterpret_cast<float *>(inputs_.at(3)->Data());
  auto variance_addr = reinterpret_cast<float *>(inputs_.at(4)->Data());
  auto output_addr = reinterpret_cast<float *>(outputs_.at(0)->Data());

  FusedBatchNorm(input_addr, scale_addr, offest_addr, mean_addr, variance_addr, input_shape_,
                 fused_batchnorm_param_->epsilon_, output_addr);
  return RET_OK;
}

kernel::LiteKernel *CpuFusedBatchnormKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *opParameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_FusedBatchNorm);
  FusedBatchnormCPUKernel *kernel = new (std::nothrow) FusedBatchnormCPUKernel(opParameter, inputs, outputs, ctx,
                                                                               primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new FusedBatchnormCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FusedBatchNorm, CpuFusedBatchnormKernelCreator)
}  // namespace mindspore::kernel
