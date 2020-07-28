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
#include "src/runtime/kernel/arm/fp32/addn.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/arithmetic.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AddN;

namespace mindspore::kernel {
namespace {
constexpr int kLeastInputNum = 2;
}

int AddNCPUKernel::Init() { return RET_OK; }

int AddNCPUKernel::ReSize() { return RET_OK; }

int AddNCPUKernel::Run() {
  auto input0_data = reinterpret_cast<float *>(inputs_[0]->Data());
  auto input1_data = reinterpret_cast<float *>(inputs_[1]->Data());
  auto output_data = reinterpret_cast<float *>(outputs_[0]->Data());
  auto element_num = inputs_[0]->ElementsNum();

  ElementAdd(input0_data, input1_data, output_data, element_num);
  for (int i = 2; i < inputs_.size(); ++i) {
    ElementAdd(reinterpret_cast<float *>(inputs_[i]->Data()), output_data, output_data, element_num);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuAddNFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const lite::Context *ctx,
                                             const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) AddNCPUKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new AddNCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed! name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, PrimitiveType_AddN, CpuAddNFp32KernelCreator)
}  // namespace mindspore::kernel

