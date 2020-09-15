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
#include "src/runtime/kernel/arm/fp32_grad/depend.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Depend;

namespace mindspore::kernel {

int DependCPUKernel::Init() { return RET_OK; }

int DependCPUKernel::ReSize() { return 0; }

int DependCPUKernel::Run() {
#if 0
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto in = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto out = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());

  memcpy(out, in, in_tensors_.at(0)->Size());
#endif
  return RET_OK;
}

kernel::LiteKernel *CpuDependFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Depend);
  auto *kernel = new (std::nothrow) DependCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);

  auto ret = kernel->Init();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Depend, CpuDependFp32KernelCreator)
}  // namespace mindspore::kernel
