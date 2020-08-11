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

#include "src/runtime/kernel/arm/fp32/unique.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Unique;

namespace mindspore::kernel {
int UniqueCPUKernel::Init() { return RET_OK; }

int UniqueCPUKernel::ReSize() { return RET_OK; }

int UniqueCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto input = reinterpret_cast<float *>(in_tensors_.at(0)->Data());
  auto output0 = reinterpret_cast<float *>(out_tensors_.at(0)->Data());
  auto output1 = reinterpret_cast<int *>(out_tensors_.at(1)->Data());

  int output0_len = 0;
  Unique(input, in_tensors_.at(0)->ElementsNum(), output0, &output0_len, output1);

  std::vector<int> out_shape = out_tensors_.at(0)->shape();
  out_shape[out_shape.size() - 1] = output0_len;
  out_tensors_.at(0)->set_shape(out_shape);
  return RET_OK;
}

kernel::LiteKernel *CpuUniqueFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *parameter, const lite::Context *ctx, const KernelKey &desc,
                                               const lite::Primitive *primitive) {
  MS_ASSERT(parameter);
  MS_ASSERT(desc.type == PrimitiveType_Unique);
  auto *kernel = new (std::nothrow) UniqueCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, name: " << parameter->name_;
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Unique, CpuUniqueFp32KernelCreator)
}  // namespace mindspore::kernel
