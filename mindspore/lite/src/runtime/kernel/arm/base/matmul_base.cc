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
#include "src/runtime/kernel/arm/base/matmul_base.h"
#include "src/runtime/kernel/arm/fp32/matmul.h"
#include "src/runtime/kernel/arm/int8/matmul_int8.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"
#include "include/context.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {
kernel::LiteKernel *CpuMatmulKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                           const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *opParameter,
                                           const lite::Context *ctx, const kernel::KernelKey &desc,
                                           const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Concat);
  auto input_tensor = inputs.at(kInputIndex);
  auto data_type = input_tensor->data_type();
  kernel::LiteKernel *kernel = nullptr;
  switch (data_type) {
    case kNumberTypeInt8:
    case kNumberTypeUInt8: {
      kernel = new (std::nothrow) MatmulInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
      if (!kernel) {
        MS_LOG(ERROR) << "kernel is nullptr.";
        return nullptr;
      }
      break;
    }

    case kNumberTypeFloat32: {
      kernel = new (std::nothrow) MatmulCPUKernel(opParameter, inputs, outputs, ctx, primitive);
      if (kernel == nullptr) {
        MS_LOG(ERROR) << "kernel is nullptr.";
        return nullptr;
      }
      break;
    }

    default:
      break;
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MatMul, CpuMatmulKernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_MatMul, CpuMatmulKernelCreator)
}  // namespace mindspore::kernel
