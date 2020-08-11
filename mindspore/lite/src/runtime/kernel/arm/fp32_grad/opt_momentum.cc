
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

#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32_grad/opt_momentum.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_OptMomentum;

namespace mindspore::kernel {

int OptMomentumCPUKernel::ReSize() { return 0; }

int OptMomentumCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  if (inputs_.size() != 5 || !outputs_.empty()) {
    MS_LOG(ERROR) << "OptMomentumCPUKernel error input output size!";
    return RET_ERROR;
  }

  if (inputs_[0]->ElementsNum() != inputs_[1]->ElementsNum() ||
      inputs_[0]->ElementsNum() != inputs_[3]->ElementsNum()) {
    MS_LOG(ERROR) << "error input data size!";
    return RET_ERROR;
  }
  auto weight = reinterpret_cast<float *>(inputs_[0]->Data());
  auto accumulate = reinterpret_cast<float *>(inputs_[1]->Data());
  float learning_rate = reinterpret_cast<float *>(inputs_[2]->Data())[0];
  auto gradient = reinterpret_cast<float *>(inputs_[3]->Data());
  float moment = reinterpret_cast<float *>(inputs_[4]->Data())[0];
  size_t elem_num = inputs_[0]->ElementsNum();
  for (size_t i = 0; i < elem_num; ++i) {
    accumulate[i] = accumulate[i] * moment + gradient[i];
    weight[i] -= accumulate[i] * learning_rate;
  }
  return RET_OK;
}

int OptMomentumCPUKernel::Init() { return 0; }

kernel::LiteKernel *CpuOptMomentumFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                    const std::vector<lite::tensor::Tensor *> &outputs,
                                                    OpParameter *opParameter, const lite::Context *ctx,
                                                    const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_OptMomentum);
  auto *kernel = new (std::nothrow) OptMomentumCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);

  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_OptMomentum, CpuOptMomentumFp32KernelCreator)
}  // namespace mindspore::kernel
