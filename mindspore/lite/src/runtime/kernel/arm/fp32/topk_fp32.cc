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

#include "src/runtime/kernel/arm/fp32/topk_fp32.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TopKFusion;

namespace mindspore::kernel {
int TopKCPUKernel::Init() {
  topk_param_->topk_node_list_ = nullptr;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TopKCPUKernel::ReSize() {
  lite::Tensor *input = in_tensors_.at(0);
  topk_param_->last_dim_size_ = input->shape().at(input->shape().size() - 1);
  topk_param_->loop_num_ = 1;
  for (size_t i = 0; i < input->shape().size() - 1; ++i) {
    topk_param_->loop_num_ *= input->shape().at(i);
  }
  return RET_OK;
}

int TopKCPUKernel::Run() {
  auto input_data = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  MS_ASSERT(input_data);
  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output_data);
  auto output_index = reinterpret_cast<int32_t *>(out_tensors_.at(1)->MutableData());
  MS_ASSERT(output_index);

  MS_ASSERT(context_->allocator != nullptr);
  if (in_tensors_.size() == 2) {
    auto input_k = reinterpret_cast<int *>(in_tensors_.at(1)->MutableData());
    topk_param_->k_ = input_k[0];
  }
  if (topk_param_->k_ > in_tensors_.at(0)->ElementsNum()) {
    MS_LOG(ERROR) << "The k value is out of the data size range.";
    return RET_ERROR;
  }
  topk_param_->topk_node_list_ = context_->allocator->Malloc(sizeof(TopkNode) * topk_param_->last_dim_size_);
  if (topk_param_->topk_node_list_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  Topk(input_data, output_data, output_index, reinterpret_cast<TopkParameter *>(op_parameter_));
  context_->allocator->Free(topk_param_->topk_node_list_);
  topk_param_->topk_node_list_ = nullptr;
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TopKFusion, LiteKernelCreator<TopKCPUKernel>)
}  // namespace mindspore::kernel
