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
#include <vector>
#include "src/kernel_registry.h"
#include "nnacl/fp32/space_to_batch_fp32.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SpaceToBatch;
using mindspore::schema::PrimitiveType_SpaceToBatchND;

namespace mindspore::kernel {
int SpaceToBatchCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SpaceToBatchCPUKernel::ReSize() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  if (input_tensor->format() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_batch only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->op_parameter_);

  for (size_t i = 0; i < DIMENSION_4D; i++) {
    param->input_shape_[i] = input_tensor->shape().at(i);
    param->output_shape_[i] = output_tensor->shape().at(i);
  }
  for (int i = 0; i < DIMENSION_4D; ++i) {
    if (param->paddings_[i] != 0) {
      param->need_paddings_ = true;
      break;
    }
  }
  if (param->need_paddings_) {
    int padding_left = 0;
    int padding_right = 0;
    if (param->m_ == 2) {
      padding_left = param->paddings_[2];
      padding_right = param->paddings_[3];
    }
    param->padded_in_shape_[kNHWC_N] = input_tensor->shape().at(kNHWC_N);
    param->padded_in_shape_[kNHWC_H] = input_tensor->shape().at(kNHWC_H) + param->paddings_[0] + param->paddings_[1];
    param->padded_in_shape_[kNHWC_W] = input_tensor->shape().at(kNHWC_W) + padding_left + padding_right;
    param->padded_in_shape_[kNHWC_C] = input_tensor->shape().at(kNHWC_C);
    param->padded_input_element_num = param->padded_in_shape_[kNHWC_N] * param->padded_in_shape_[kNHWC_H] *
                                      param->padded_in_shape_[kNHWC_W] * param->padded_in_shape_[kNHWC_C];
  }
  return RET_OK;
}

int SpaceToBatchCPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  auto input_ptr = reinterpret_cast<const float *>(input_tensor->MutableData());
  auto output_ptr = reinterpret_cast<float *>(output_tensor->MutableData());
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->op_parameter_);

  if (param->need_paddings_) {
    padded_input_ = context_->allocator->Malloc(param->padded_input_element_num * sizeof(float));
    if (padded_input_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
    auto padded_input = reinterpret_cast<float *>(padded_input_);
    DoSpaceToBatchPaddingNHWC(input_ptr, padded_input, param->input_shape_, param->paddings_, param->padded_in_shape_);
    DoSpaceToBatchNHWC(padded_input, output_ptr, param->block_sizes_, param->padded_in_shape_, param->output_shape_);
    FreeTmpBuffer();
  } else {
    DoSpaceToBatchNHWC(input_ptr, output_ptr, param->block_sizes_, param->input_shape_, param->output_shape_);
  }
  return RET_OK;
}

void SpaceToBatchCPUKernel::FreeTmpBuffer() {
  if (padded_input_ != nullptr) {
    context_->allocator->Free(padded_input_);
    padded_input_ = nullptr;
  }
}

kernel::LiteKernel *CpuSpaceToBatchFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                     const std::vector<lite::Tensor *> &outputs, OpParameter *param,
                                                     const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                     const mindspore::lite::PrimitiveC *primitive) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "Input param is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) SpaceToBatchCPUKernel(param, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SpaceToBatchCPUKernel fail!";
    free(param);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << param->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(param->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToBatch, CpuSpaceToBatchFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToBatchND, CpuSpaceToBatchFp32KernelCreator)
}  // namespace mindspore::kernel
