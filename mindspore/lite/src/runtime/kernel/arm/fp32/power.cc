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

#include "src/runtime/kernel/arm/fp32/power.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Power;

namespace mindspore::kernel {
int PowerCPUKernel::Init() { return RET_OK; }

int PowerCPUKernel::ReSize() { return RET_OK; }

int PowerImpl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto kernel = reinterpret_cast<PowerCPUKernel *>(cdata);
  auto ret = kernel->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerImpl error: " << ret;
    return ret;
  }
  return RET_OK;
}

int PowerCPUKernel::Run() {
  int ret = LiteBackendParallelLaunch(PowerImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerCPUKernel error: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int PowerCPUKernel::RunImpl(int task_id) {
  auto input_addr = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto output_addr = reinterpret_cast<float *>(outputs_.at(0)->Data());
  auto size = inputs_.at(0)->Size();
  int stride = UP_DIV(size, thread_count_);
  int len = MSMIN(stride, size - stride * task_id);

  Power(input_addr + stride * task_id, output_addr + stride * task_id, len, power_, scale_, shift_);
  return RET_OK;
}

kernel::LiteKernel *CpuPowerFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Power);
  auto *kernel =
    new (std::nothrow) PowerCPUKernel(reinterpret_cast<PowerParameter *>(opParameter), inputs, outputs, ctx);
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
  }
  return kernel;
}

REG_KERNEL(kCPU, PrimitiveType_Power, CpuPowerFp32KernelCreator)
}  // namespace mindspore::kernel
