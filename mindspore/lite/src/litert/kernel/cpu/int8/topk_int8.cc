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

#include "src/litert/kernel/cpu/int8/topk_int8.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TopKFusion;

namespace mindspore::kernel {
int TopKInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C1NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TopKInt8CPUKernel::ReSize() {
  TopkParameter *parameter = reinterpret_cast<TopkParameter *>(op_parameter_);
  CHECK_NULL_RETURN(parameter);
  lite::Tensor *input = in_tensors_.at(0);
  parameter->dim_size_ = input->shape().at(static_cast<size_t>(parameter->axis_));
  parameter->outer_loop_num_ = 1;
  for (size_t i = 0; i < static_cast<size_t>(parameter->axis_); ++i) {
    parameter->outer_loop_num_ *= input->shape().at(i);
  }
  parameter->inner_loop_num_ = 1;
  for (size_t i = static_cast<size_t>(parameter->axis_ + 1); i < input->shape().size(); ++i) {
    parameter->inner_loop_num_ *= input->shape().at(i);
  }
  CHECK_LESS_RETURN(parameter->dim_size_, parameter->k_);
  return RET_OK;
}

int TopKInt8CPUKernel::Run() {
  int8_t *input_data = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(input_data);
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(output_data);
  int32_t *output_index = reinterpret_cast<int32_t *>(out_tensors_.at(1)->data());
  CHECK_NULL_RETURN(output_index);

  MS_ASSERT(ms_context_->allocator != nullptr);
  TopkParameter *parameter = reinterpret_cast<TopkParameter *>(op_parameter_);
  parameter->topk_node_list_ = ms_context_->allocator->Malloc(sizeof(TopkNodeInt8) * parameter->dim_size_);
  if (parameter->topk_node_list_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  TopkInt8(input_data, output_data, output_index, reinterpret_cast<TopkParameter *>(op_parameter_));
  ms_context_->allocator->Free(parameter->topk_node_list_);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_TopKFusion, LiteKernelCreator<TopKInt8CPUKernel>)
}  // namespace mindspore::kernel
