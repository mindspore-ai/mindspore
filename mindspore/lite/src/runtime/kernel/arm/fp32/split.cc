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

#include <string.h>
#include <vector>
#include "src/runtime/kernel/arm/fp32/split.h"
#include "src/runtime/kernel/arm/opclib/split.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore::kernel {

int SplitCPUKernel::Init() {
  auto in_tensor = inputs_.front();
  input_ptr_ = reinterpret_cast<float *>(in_tensor->Data());
  auto input_shape = in_tensor->shape();
  auto param = reinterpret_cast<SplitParameter *>(opParameter);

  param->strides_[input_shape.size() - 1] = 1;
  for (int i = input_shape.size() - 2; i >= 0; i--) {
    param->strides_[i] = param->strides_[i + 1] * input_shape[i + 1];
  }

  param->split_count_ =
    param->strides_[0] * input_shape[0] / (input_shape[param->split_dim_] * param->strides_[param->split_dim_]);
  for (int i = 0; i < param->num_split_; i++) {
    output_ptr_.push_back(reinterpret_cast<float *>(outputs_.at(i)->Data()));
  }
  param->n_dims_ = input_shape.size();

  if (param->split_sizes_[0] == 0) {
    if (input_shape[param->split_dim_] % param->num_split_ != 0) {
      MS_LOG(ERROR) << "Default split size is not usable.";
      return RET_ERROR;
    }
    int split_size = input_shape[param->split_dim_] / param->num_split_;
    for (int i = 0; i < param->num_split_; i++) {
      param->split_sizes_[i] = split_size;
    }
  }

  num_unit_ = param->split_count_ * param->num_split_;
  unit_size_ = param->strides_[param->split_dim_];
  thread_n_num_ = MSMIN(thread_num_, num_unit_);
  thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);
  return RET_OK;
}

int SplitCPUKernel::ReSize() { return RET_OK; }

int SplitCPUKernel::Split(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  auto ret = DoSplit(input_ptr_, output_ptr_.data(), inputs_.front()->shape().data(), thread_offset, num_unit_thread,
                     reinterpret_cast<SplitParameter *>(opParameter));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Split error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<SplitCPUKernel *>(cdata);
  auto ret = g_kernel->Split(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SplitRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitCPUKernel::Run() {
  int ret = LiteBackendParallelLaunch(SplitRun, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuSplitFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Split);
  auto *kernel = new (std::nothrow) SplitCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Split, CpuSplitFp32KernelCreator)
}  // namespace mindspore::kernel
