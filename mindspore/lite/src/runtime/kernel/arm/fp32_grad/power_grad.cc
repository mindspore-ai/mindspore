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

#include "src/runtime/kernel/arm/fp32_grad/power_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/nnacl/fp32/arithmetic.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PowerGrad;

namespace mindspore::kernel {
int PowerGradCPUKernel::Init() { return RET_OK; }

int PowerGradCPUKernel::ReSize() { return RET_OK; }

int PowerGradCPUKernel::Run() {
  auto dy_addr = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto x_addr = reinterpret_cast<float *>(inputs_.at(1)->Data());
  auto dx_addr = reinterpret_cast<float *>(outputs_.at(0)->Data());
  auto size = inputs_.at(0)->ElementsNum();

  float exp = power_ - 1;
  Power(x_addr, &exp, dx_addr, size, scale_, shift_, true);
  ElementMul(dx_addr, dy_addr, dx_addr, size);
  float scale = scale_ * power_;
  for (int i = 0; i < size; i++) {
    dx_addr[i] *= scale;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuPowerGradFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *opParameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_PowerGrad);
  auto *kernel = new (std::nothrow) PowerGradCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PowerGrad, CpuPowerGradFp32KernelCreator)
}  // namespace mindspore::kernel
