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

#include "src/runtime/kernel/arm/fp32/unstack.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Unstack;

namespace mindspore::kernel {
int UnstackCPUKernel::Init() {
  auto input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  size_t shape_size = input->shape().size();

  auto para = reinterpret_cast<UnstackParameter *>(opParameter);
  para->pre_dims_ = 1;
  para->axis_dim_ = 1;
  para->after_dims_ = 1;
  if (para->axis_ < 0) {
    para->axis_ += shape_size;
  }
  for (size_t i = 0; i < shape_size; i++) {
    if (i < para->axis_) {
      para->pre_dims_ *= input->DimensionSize(i);
    } else if (i > para->axis_) {
      para->after_dims_ *= input->DimensionSize(i);
    } else {
      para->axis_dim_ = input->DimensionSize(i);
    }
  }

  output_addr_array_ = reinterpret_cast<float **>(malloc(sizeof(float *) * outputs_.size()));
  if (output_addr_array_ == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc memory";
    return lite::RET_ERROR;
  }
  return RET_OK;
}

int UnstackCPUKernel::ReSize() { return RET_OK; }

int UnstackCPUKernel::Run() {
  float *input = reinterpret_cast<float *>(inputs_.at(0)->Data());
  size_t out_num = outputs_.size();
  for (size_t i = 0; i < out_num; i++) {
    output_addr_array_[i] = reinterpret_cast<float *>(outputs_.at(i)->Data());
  }
  Unistack(input, output_addr_array_, reinterpret_cast<UnstackParameter *>(opParameter));
  return RET_OK;
}

kernel::LiteKernel *CpuUnstackFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                const std::vector<lite::tensor::Tensor *> &outputs,
                                                OpParameter *parameter, const lite::Context *ctx,
                                                const KernelKey &desc) {
  MS_ASSERT(parameter != nullptr);
  MS_ASSERT(desc.type == PrimitiveType_Unstack);
  auto *kernel = new (std::nothrow) UnstackCPUKernel(parameter, inputs, outputs);
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

REG_KERNEL(kCPU, PrimitiveType_Unstack, CpuUnstackFp32KernelCreator)
}  // namespace mindspore::kernel

