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
  auto input_tensor = in_tensors_.at(FIRST_INPUT);
  auto output_tensor = out_tensors_.at(FIRST_INPUT);
  for (size_t i = 0; i < DIMENSION_4D; i++) {
    param_->input_shape_[i] = input_tensor->shape().at(i);
    param_->output_shape_[i] = output_tensor->shape().at(i);
  }
  ComputeStrides(param_->input_shape_, param_->in_stride_, DIMENSION_4D);
  ComputeStrides(param_->output_shape_, param_->out_stride_, DIMENSION_4D);
  auto block_shape_data = in_tensors_.at(SECOND_INPUT)->data();
  auto block_shape = static_cast<int *>(block_shape_data);
  MS_ASSERT(block_shape != nullptr);
  for (int i = 0; i < in_tensors_.at(SECOND_INPUT)->ElementsNum(); i++) {
    param_->block_sizes_[i] = block_shape[i];
  }
  auto padding_data = in_tensors_.at(THIRD_INPUT)->data();
  auto padding = static_cast<int *>(padding_data);
  MS_ASSERT(padding != nullptr);
  for (int i = 0; i < in_tensors_.at(THIRD_INPUT)->ElementsNum(); i++) {
    param_->paddings_[i] = padding[i];
  }
}

int SpaceToBatchCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SpaceToBatchFp32Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto op = reinterpret_cast<SpaceToBatchCPUKernel *>(cdata);
  auto ret = op->DoRun(task_id);
  return ret;
}

int SpaceToBatchCPUKernel::ReSize() {
  if (in_tensors_.size() == DIMENSION_3D) {
    if (in_tensors_.at(SECOND_INPUT) != nullptr && in_tensors_.at(SECOND_INPUT)->IsConst() &&
        in_tensors_.at(THIRD_INPUT) != nullptr && in_tensors_.at(THIRD_INPUT)->IsConst()) {
      ProcessInput();
    }
  }
  auto input_tensor = in_tensors_.at(FIRST_INPUT);
  auto output_tensor = out_tensors_.at(FIRST_INPUT);
  for (size_t i = 0; i < DIMENSION_4D; i++) {
    param_->input_shape_[i] = input_tensor->shape().at(i);
    param_->output_shape_[i] = output_tensor->shape().at(i);
  }

  ComputeStrides(param_->input_shape_, param_->in_stride_, DIMENSION_4D);
  ComputeStrides(param_->output_shape_, param_->out_stride_, DIMENSION_4D);
  return RET_OK;
}

int SpaceToBatchCPUKernel::DoRun(int task_id) {
  auto ret =
    DoSpaceToBatch(input_ptr_, output_ptr_, param_->input_shape_, param_->output_shape_, param_->in_stride_,
                   param_->out_stride_, param_->block_sizes_, param_->paddings_, op_parameter_->thread_num_, task_id);
  return ret;
}

int SpaceToBatchCPUKernel::Run() {
  input_ptr_ = reinterpret_cast<float *>(in_tensors_.at(FIRST_INPUT)->data());
  MS_ASSERT(input_ptr_ != nullptr);
  output_ptr_ = reinterpret_cast<float *>(out_tensors_.at(FIRST_INPUT)->data());
  MS_ASSERT(output_ptr_ != nullptr);
  if (in_tensors_.size() == DIMENSION_3D) {
    if (!in_tensors_.at(SECOND_INPUT)->IsConst() || !in_tensors_.at(THIRD_INPUT)->IsConst()) {
      ProcessInput();
    }
  }

  ParallelLaunch(this->ms_context_, SpaceToBatchFp32Run, this, op_parameter_->thread_num_);

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToBatch, LiteKernelCreator<SpaceToBatchCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToBatchND, LiteKernelCreator<SpaceToBatchCPUKernel>)
}  // namespace mindspore::kernel
