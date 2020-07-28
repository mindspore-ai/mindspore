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
  lite::tensor::Tensor *input = inputs_.at(0);
  topk_parameter_->last_dim_size_ = input->shape()[input->shape().size() - 1];
  topk_parameter_->loop_num_ = 1;
  for (int i = 0; i < input->shape().size() - 1; ++i) {
    topk_parameter_->loop_num_ *= input->shape()[i];
  }
  return RET_OK;
}

int TopKCPUKernel::ReSize() { return RET_OK; }

int TopKCPUKernel::Run() {
  auto input_data = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto output_data = reinterpret_cast<float *>(outputs_.at(0)->Data());
  auto output_index = reinterpret_cast<float *>(outputs_.at(1)->Data());

  Node *top_map = reinterpret_cast<Node *>(malloc(sizeof(Node) * topk_parameter_->last_dim_size_));
  MS_EXCEPTION_IF_NULL(top_map);
  topk_parameter_->topk_node_list_ = top_map;
  Topk(input_data, output_data, output_index, topk_parameter_);
  free(top_map);
  topk_parameter_->topk_node_list_ = nullptr;
  return RET_OK;
}

kernel::LiteKernel *CpuTopKFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *parameter,
                                             const lite::Context *ctx, const KernelKey &desc) {
  MS_EXCEPTION_IF_NULL(parameter);
  MS_ASSERT(desc.type == PrimitiveType_Tile);
  auto *kernel = new (std::nothrow) TopKCPUKernel(parameter, inputs, outputs);
  MS_EXCEPTION_IF_NULL(kernel);

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, PrimitiveType_TopK, CpuTopKFp32KernelCreator)
}  // namespace mindspore::kernel

