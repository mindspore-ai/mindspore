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

#include "src/runtime/kernel/arm/fp32/unstack_fp32.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Unstack;

namespace mindspore::kernel {
int UnstackCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int UnstackCPUKernel::ReSize() {
  auto input = in_tensors_.at(0);
  MS_ASSERT(input != nullptr);
  size_t shape_size = input->shape().size();

  auto para = reinterpret_cast<UnstackParameter *>(op_parameter_);
  para->pre_dims_ = 1;
  para->axis_dim_ = 1;
  para->after_dims_ = 1;
  if (para->axis_ < 0) {
    para->axis_ += shape_size;
  }

  for (size_t i = 0; i < shape_size; i++) {
    if (static_cast<int>(i) < para->axis_) {
      para->pre_dims_ *= input->DimensionSize(i);
    } else if (static_cast<int>(i) > para->axis_) {
      para->after_dims_ *= input->DimensionSize(i);
    } else {
      para->axis_dim_ = input->DimensionSize(i);
    }
  }
  if (output_addr_array_ != nullptr) {
    free(output_addr_array_);
    output_addr_array_ = nullptr;
  }
  output_addr_array_ = reinterpret_cast<void **>(malloc(sizeof(void *) * out_tensors_.size()));
  if (output_addr_array_ == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc memory";
    return lite::RET_ERROR;
  }
  return RET_OK;
}

int UnstackCPUKernel::Run() {
  float *input = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  MS_ASSERT(input);
  size_t out_num = out_tensors_.size();
  for (size_t i = 0; i < out_num; i++) {
    output_addr_array_[i] = out_tensors_.at(i)->data_c();
  }
  MS_ASSERT(output_addr_array_);
  auto para = reinterpret_cast<UnstackParameter *>(op_parameter_);
  para->num_ = out_num;
  Unstack(input, output_addr_array_, para, sizeof(float));
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Unstack, LiteKernelCreator<UnstackCPUKernel>)
}  // namespace mindspore::kernel
