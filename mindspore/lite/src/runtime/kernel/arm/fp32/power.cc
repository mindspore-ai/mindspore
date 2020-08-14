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
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto ret = LiteBackendParallelLaunch(PowerImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerCPUKernel error: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int PowerCPUKernel::RunImpl(int task_id) {
  auto x_addr = reinterpret_cast<float *>(in_tensors_[0]->Data());
  auto output_addr = reinterpret_cast<float *>(out_tensors_[0]->Data());
  auto size = in_tensors_[0]->ElementsNum();
  int stride = UP_DIV(size, thread_count_);
  int len = MSMIN(stride, size - stride * task_id);
  float *exp_addr = nullptr;
  bool broadcast = true;
  if (in_tensors_.size() == 2) {
    exp_addr = reinterpret_cast<float *>(in_tensors_[1]->Data());
    broadcast = false;
  }
  float *cur_exp = nullptr;
  if (broadcast) {
    cur_exp = &power_;
  } else {
    cur_exp = exp_addr + stride * task_id;
  }
  Power(x_addr + stride * task_id, cur_exp, output_addr + stride * task_id, len, scale_, shift_, broadcast);
  return RET_OK;
}

kernel::LiteKernel *CpuPowerFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Power);
  PowerCPUKernel *kernel = new (std::nothrow) PowerCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PowerCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Power, CpuPowerFp32KernelCreator)
}  // namespace mindspore::kernel
