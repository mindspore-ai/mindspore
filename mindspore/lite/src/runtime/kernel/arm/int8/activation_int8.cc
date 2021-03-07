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

#include "src/runtime/kernel/arm/int8/relux_int8.h"
#include "src/runtime/kernel/arm/int8/hswish_int8.h"
#include "src/runtime/kernel/arm/int8/sigmoid_int8.h"
#include "src/runtime/kernel/arm/int8/tanh_int8.h"
#include "src/runtime/kernel/arm/int8/leaky_relu_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {
kernel::LiteKernel *CpuActivationInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                                   const lite::InnerContext *ctx, const KernelKey &desc) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr";
    return nullptr;
  }
  MS_ASSERT(inputs.at(0));
  auto type = (reinterpret_cast<ActivationParameter *>(parameter))->type_;
  kernel::LiteKernel *kernel = nullptr;
  switch (static_cast<schema::ActivationType>(type)) {
    case schema::ActivationType_RELU:
      kernel = new (std::nothrow) ReluInt8CPUKernel(parameter, inputs, outputs, ctx);
      break;
    case schema::ActivationType_RELU6:
      kernel = new (std::nothrow) Relu6Int8CPUKernel(parameter, inputs, outputs, ctx);
      break;
    case schema::ActivationType_HSWISH:
      kernel = new (std::nothrow) HswishInt8CPUKernel(parameter, inputs, outputs, ctx);
      break;
    case schema::ActivationType_SIGMOID:
      kernel = new (std::nothrow) SigmoidInt8CPUKernel(parameter, inputs, outputs, ctx);
      break;
    case schema::ActivationType_LEAKY_RELU:
      kernel = new (std::nothrow) LeakyReluInt8CPUKernel(parameter, inputs, outputs, ctx);
      break;
    case schema::ActivationType_TANH:
      kernel = new (std::nothrow) TanhInt8CPUKernel(parameter, inputs, outputs, ctx);
      break;
    default:
      break;
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed";
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Activation, CpuActivationInt8KernelCreator)
}  // namespace mindspore::kernel
