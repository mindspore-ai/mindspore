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

#include "src/runtime/kernel/arm/fp32_grad/smooth_l1_loss.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SmoothL1Loss;

namespace mindspore::kernel {

int SmoothL1LossCPUKernel::ReSize() { return RET_OK; }

int SmoothL1LossCPUKernel::Execute(int task_id) {
  SmoothL1LossParameter *smooth_l1_loss_param = reinterpret_cast<SmoothL1LossParameter *>(op_parameter_);

  auto predict = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto target = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto *out = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());

  const size_t length = in_tensors_.at(0)->ElementsNum();

  size_t stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);

  size_t start = stride * task_id;
  size_t end = start + count;

  const float zero = 0.0f;
  const float half = 0.5f;
  const float beta = smooth_l1_loss_param->beta_;

  for (uint64_t i = start; i < end; ++i) {
    float diff = predict[i] - target[i];
    if (diff < zero) {
      diff = -diff;
    }
    if (diff < beta) {
      out[i] = half * diff * diff / beta;
    } else {
      out[i] = diff - (half * beta);
    }
  }

  return RET_OK;
}

int SmoothL1LossRun(void *cdata, int task_id) {
  auto smooth_l1_loss_kernel = reinterpret_cast<SmoothL1LossCPUKernel *>(cdata);
  auto error_code = smooth_l1_loss_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SmoothL1Loss error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SmoothL1LossCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, SmoothL1LossRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SmoothL1Loss function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SmoothL1LossCPUKernel::Init() { return RET_OK; }

kernel::LiteKernel *CpuSmoothL1LossFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                     const std::vector<lite::Tensor *> &outputs,
                                                     OpParameter *opParameter, const lite::InnerContext *ctx,
                                                     const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_SmoothL1Loss);
  auto *kernel = new (std::nothrow) SmoothL1LossCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SmoothL1Loss failed";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SmoothL1Loss, CpuSmoothL1LossFp32KernelCreator)
}  // namespace mindspore::kernel
