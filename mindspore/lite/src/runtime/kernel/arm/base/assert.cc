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

#include "src/runtime/kernel/arm/base/assert.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Assert;

namespace mindspore::kernel {

int AssertCPUKernel::Init() { return RET_OK; }

int AssertCPUKernel::ReSize() { return RET_OK; }

int AssertCPUKernel::Run() {
  auto cond = reinterpret_cast<bool *>(in_tensors_.front()->data_c());
  if (*cond) {
    return RET_OK;
  } else {
    for (size_t i = 1; i < in_tensors_.size(); i++) {
      MS_LOG(ERROR) << in_tensors_.at(i)->ToString();
    }
    return RET_ERROR;
  }
}

kernel::LiteKernel *CpuAssertKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                           const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                           const lite::InnerContext *ctx, const KernelKey &desc,
                                           const mindspore::lite::PrimitiveC *primitive) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "ctx is nullptr";
    free(parameter);
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_Assert);
  auto *kernel = new (std::nothrow) AssertCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, name: " << parameter->name_;
    free(parameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Assert, CpuAssertKernelCreator)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Assert, CpuAssertKernelCreator)
}  // namespace mindspore::kernel
