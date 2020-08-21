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

#include "src/runtime/kernel/arm/fp32/topk.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TopK;

namespace mindspore::kernel {
int TopKCPUKernel::Init() {
  TopkParameter *parameter = reinterpret_cast<TopkParameter *>(op_parameter_);
  parameter->topk_node_list_ = nullptr;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TopKCPUKernel::ReSize() {
  lite::tensor::Tensor *input = in_tensors_.at(0);
  TopkParameter *parameter = reinterpret_cast<TopkParameter *>(op_parameter_);
  parameter->last_dim_size_ = input->shape()[input->shape().size() - 1];
  parameter->loop_num_ = 1;
  for (size_t i = 0; i < input->shape().size() - 1; ++i) {
    parameter->loop_num_ *= input->shape()[i];
  }
  return RET_OK;
}

int TopKCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto input_data = reinterpret_cast<float *>(in_tensors_.at(0)->Data());
  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->Data());
  auto output_index = reinterpret_cast<int32_t *>(out_tensors_.at(1)->Data());

  MS_ASSERT(context_->allocator != nullptr);
  TopkParameter *parameter = reinterpret_cast<TopkParameter *>(op_parameter_);
  parameter->topk_node_list_ = context_->allocator->Malloc(sizeof(TopkNode) * parameter->last_dim_size_);
  if (parameter->topk_node_list_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  Topk(input_data, output_data, output_index, reinterpret_cast<TopkParameter *>(op_parameter_));
  context_->allocator->Free(parameter->topk_node_list_);
  return RET_OK;
}

kernel::LiteKernel *CpuTopKFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *parameter,
                                             const lite::Context *ctx, const KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_Tile);
  auto *kernel = new (std::nothrow) TopKCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new TopKCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TopK, CpuTopKFp32KernelCreator)
}  // namespace mindspore::kernel
