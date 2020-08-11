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
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  TopkParameter *parameter = reinterpret_cast<TopkParameter *>(op_parameter_);
  lite::tensor::Tensor *input = in_tensors_.at(0);
  parameter->last_dim_size_ = input->shape()[input->shape().size() - 1];
  parameter->loop_num_ = 1;
  for (int i = 0; i < input->shape().size() - 1; ++i) {
    parameter->loop_num_ *= input->shape()[i];
  }

  parameter->topk_node_list_ = malloc(sizeof(TopkNode) * parameter->last_dim_size_);
  if (parameter->topk_node_list_ == nullptr) {
    MS_LOG(ERROR) << "malloc fail.";
    return RET_ERROR;
  }
  return RET_OK;
}

int TopKCPUKernel::ReSize() { return RET_OK; }

int TopKCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto input_data = reinterpret_cast<float *>(in_tensors_.at(0)->Data());
  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->Data());
  auto output_index = reinterpret_cast<int32_t *>(out_tensors_.at(1)->Data());

  Topk(input_data, output_data, output_index, reinterpret_cast<TopkParameter *>(op_parameter_));
  return RET_OK;
}

kernel::LiteKernel *CpuTopKFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *parameter,
                                             const lite::Context *ctx, const KernelKey &desc,
                                             const lite::Primitive *primitive) {
  MS_ASSERT(parameter != nullptr);
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
