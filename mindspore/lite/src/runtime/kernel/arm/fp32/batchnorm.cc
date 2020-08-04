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

#include "src/runtime/kernel/arm/fp32/batchnorm.h"
#include <cmath>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNorm;

namespace mindspore::kernel {
int BatchnormCPUKernel::Init() { return RET_OK; }

int BatchnormCPUKernel::ReSize() { return RET_OK; }

int BatchnormCPUKernel::DoExecute(int tid) {
  int count = MSMIN(thread_unit_, units_ - tid * thread_unit_);
  if (count <= 0) {
    return RET_OK;
  }
  int offset = tid * thread_unit_ * channel_;
  BatchNorm(in_addr_ + offset, mean_addr_, var_addr_, count, channel_, batchnorm_param_->epsilon_, out_addr_ + offset);
  return RET_OK;
}

int BatchNormRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<BatchnormCPUKernel *>(cdata);
  auto ret = g_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int BatchnormCPUKernel::Run() {
  in_addr_ = reinterpret_cast<float *>(inputs_.at(0)->Data());
  mean_addr_ = reinterpret_cast<float *>(inputs_.at(1)->Data());
  var_addr_ = reinterpret_cast<float *>(inputs_.at(2)->Data());
  out_addr_ = reinterpret_cast<float *>(outputs_.at(0)->Data());
  auto input_shapes = inputs_[0]->shape();
  channel_ = input_shapes[3];
  units_ = 1;
  for (int i = 0; i < 3; i++) {
    units_ *= input_shapes[i];
  }
  thread_count_ = MSMIN(thread_count_, units_);
  thread_unit_ = UP_DIV(units_, thread_count_);
  int ret = LiteBackendParallelLaunch(BatchNormRun, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuBatchnormKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_BatchNorm);
  auto *kernel = new (std::nothrow) BatchnormCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BatchNormCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchNorm, CpuBatchnormKernelCreator)
}  // namespace mindspore::kernel
