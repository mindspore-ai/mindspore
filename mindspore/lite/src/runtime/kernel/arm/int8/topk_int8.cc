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

#include "src/runtime/kernel/arm/int8/topk_int8.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TopKFusion;

namespace mindspore::kernel {
int TopKInt8CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TopKInt8CPUKernel::ReSize() {
  TopkParameter *parameter = reinterpret_cast<TopkParameter *>(op_parameter_);
  MS_ASSERT(parameter);
  lite::Tensor *input = in_tensors_.at(0);
  MS_ASSERT(input);
  parameter->last_dim_size_ = input->shape().at(input->shape().size() - 1);
  parameter->loop_num_ = 1;
  for (size_t i = 0; i < input->shape().size() - 1; ++i) {
    parameter->loop_num_ *= input->shape().at(i);
  }
  return RET_OK;
}

int TopKInt8CPUKernel::Run() {
  int8_t *input_data = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  MS_ASSERT(input_data);
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output_data);
  int32_t *output_index = reinterpret_cast<int32_t *>(out_tensors_.at(1)->MutableData());
  MS_ASSERT(output_index);

  MS_ASSERT(context_->allocator != nullptr);
  TopkParameter *parameter = reinterpret_cast<TopkParameter *>(op_parameter_);
  parameter->topk_node_list_ = context_->allocator->Malloc(sizeof(TopkNodeInt8) * parameter->last_dim_size_);
  if (parameter->topk_node_list_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  TopkInt8(input_data, output_data, output_index, reinterpret_cast<TopkParameter *>(op_parameter_));
  context_->allocator->Free(parameter->topk_node_list_);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_TopKFusion, LiteKernelCreator<TopKInt8CPUKernel>)
}  // namespace mindspore::kernel
