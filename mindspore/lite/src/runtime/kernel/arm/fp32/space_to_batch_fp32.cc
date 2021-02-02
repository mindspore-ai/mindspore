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
#include "src/runtime/kernel/arm/fp32/space_to_batch_fp32.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SpaceToBatch;
using mindspore::schema::PrimitiveType_SpaceToBatchND;

namespace mindspore::kernel {
void SpaceToBatchCPUKernel::ProcessInput() {
  MS_ASSERT(in_tensors_[1] != nullptr);
  MS_ASSERT(in_tensors_[2] != nullptr);
  auto input_tensor = in_tensors_.at(0);
  MS_ASSERT(input_tensor);
  auto output_tensor = out_tensors_.at(0);
  MS_ASSERT(output_tensor);
  MS_ASSERT(param_);
  for (size_t i = 0; i < DIMENSION_4D; i++) {
    param_->input_shape_[i] = input_tensor->shape().at(i);
    param_->output_shape_[i] = output_tensor->shape().at(i);
  }
  ComputeStrides(param_->input_shape_, param_->in_stride_, DIMENSION_4D);
  ComputeStrides(param_->output_shape_, param_->out_stride_, DIMENSION_4D);
  auto block_shape_data = in_tensors_[1]->data_c();
  auto block_shape = static_cast<int *>(block_shape_data);
  for (int i = 0; i < in_tensors_[1]->ElementsNum(); i++) {
    param_->block_sizes_[i] = block_shape[i];
  }
  auto padding_data = in_tensors_[2]->data_c();
  auto padding = static_cast<int *>(padding_data);
  for (int i = 0; i < in_tensors_[2]->ElementsNum(); i++) {
    param_->paddings_[i] = padding[i];
  }
}

int SpaceToBatchCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SpaceToBatchFp32Run(void *cdata, int task_id) {
  auto op = reinterpret_cast<SpaceToBatchCPUKernel *>(cdata);
  op->DoRun(task_id);
  return RET_OK;
}

int SpaceToBatchCPUKernel::ReSize() {
  if (in_tensors_.size() == 3) {
    if (in_tensors_[1] != nullptr && in_tensors_[1]->IsConst() && in_tensors_[2] != nullptr &&
        in_tensors_[2]->IsConst()) {
      ProcessInput();
    }
  }
  auto input_tensor = in_tensors_.at(0);
  MS_ASSERT(input_tensor);
  auto output_tensor = out_tensors_.at(0);
  MS_ASSERT(output_tensor);
  MS_ASSERT(param_);
  for (size_t i = 0; i < DIMENSION_4D; i++) {
    param_->input_shape_[i] = input_tensor->shape().at(i);
    param_->output_shape_[i] = output_tensor->shape().at(i);
  }

  ComputeStrides(param_->input_shape_, param_->in_stride_, DIMENSION_4D);
  ComputeStrides(param_->output_shape_, param_->out_stride_, DIMENSION_4D);
  return RET_OK;
}

void SpaceToBatchCPUKernel::DoRun(int task_id) {
  DoSpaceToBatch(input_ptr_, output_ptr_, param_->input_shape_, param_->output_shape_, param_->in_stride_,
                 param_->out_stride_, param_->block_sizes_, param_->paddings_, op_parameter_->thread_num_, task_id);
  return;
}

int SpaceToBatchCPUKernel::Run() {
  MS_ASSERT(in_tensors_[0] != nullptr);
  input_ptr_ = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
  output_ptr_ = reinterpret_cast<float *>(out_tensors_.at(0)->data_c());
  if (in_tensors_.size() == 3) {
    if (!in_tensors_[1]->IsConst() || !in_tensors_[2]->IsConst()) {
      ProcessInput();
    }
  }

  ParallelLaunch(this->context_->thread_pool_, SpaceToBatchFp32Run, this, op_parameter_->thread_num_);

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToBatch, LiteKernelCreator<SpaceToBatchCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToBatchND, LiteKernelCreator<SpaceToBatchCPUKernel>)
}  // namespace mindspore::kernel
