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

#include "src/litert/kernel/cpu/fp32/topk_fp32.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TopKFusion;

namespace mindspore::kernel {
int TopKCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  topk_param_->topk_node_list_ = nullptr;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TopKCPUKernel::ReSize() {
  lite::Tensor *input = in_tensors_.at(0);
  topk_param_->dim_size_ = input->shape().at(static_cast<size_t>(topk_param_->axis_));
  topk_param_->outer_loop_num_ = 1;
  for (size_t i = 0; i < static_cast<size_t>(topk_param_->axis_); ++i) {
    topk_param_->outer_loop_num_ *= input->shape().at(i);
  }
  topk_param_->inner_loop_num_ = 1;
  for (size_t i = static_cast<size_t>(topk_param_->axis_ + 1); i < input->shape().size(); ++i) {
    topk_param_->inner_loop_num_ *= input->shape().at(i);
  }
  return RET_OK;
}

int TopKCPUKernel::Run() {
  auto input_data = in_tensors_.at(0)->data();
  CHECK_NULL_RETURN(input_data);
  auto output_data = out_tensors_.at(0)->data();
  CHECK_NULL_RETURN(output_data);
  auto output_index = reinterpret_cast<int32_t *>(out_tensors_.at(1)->data());
  CHECK_NULL_RETURN(output_index);

  if (in_tensors_.size() == C2NUM) {
    auto input_k = reinterpret_cast<int *>(in_tensors_.at(1)->data());
    CHECK_NULL_RETURN(input_k);
    topk_param_->k_ = input_k[0];
  }
  if (topk_param_->k_ > in_tensors_.at(0)->ElementsNum()) {
    MS_LOG(ERROR) << "The k value is out of the data size range.";
    return RET_ERROR;
  }
  if (topk_param_->k_ > topk_param_->dim_size_) {
    MS_LOG(ERROR) << "The k value is out of the data size range.";
    return RET_ERROR;
  }
  MS_ASSERT(ms_context_->allocator != nullptr);
  topk_param_->topk_node_list_ =
    ms_context_->allocator->Malloc(sizeof(TopkNode) * static_cast<size_t>(topk_param_->dim_size_));
  if (topk_param_->topk_node_list_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(topk_func_);
  topk_func_(input_data, output_data, output_index, reinterpret_cast<TopkParameter *>(op_parameter_));
  ms_context_->allocator->Free(topk_param_->topk_node_list_);
  topk_param_->topk_node_list_ = nullptr;
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TopKFusion, LiteKernelCreator<TopKCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TopKFusion, LiteKernelCreator<TopKCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_TopKFusion, LiteKernelCreator<TopKCPUKernel>)
#endif
}  // namespace mindspore::kernel
