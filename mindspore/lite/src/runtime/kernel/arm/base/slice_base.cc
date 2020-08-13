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
#include "src/runtime/kernel/arm/base/slice_base.h"
#include <vector>
#include "src/runtime/kernel/arm/int8/slice_int8.h"
#include "src/runtime/kernel/arm/fp32/slice.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Slice;

namespace mindspore::kernel {
int SliceBaseCPUKernel::Init() { return RET_OK; }

int SliceBaseCPUKernel::ReSize() {
  auto input_shape = in_tensors_[0]->shape();
  if (input_shape.size() > DIMENSION_4D) {
    MS_LOG(ERROR) << "input dimension num should <= " << DIMENSION_4D;
    return RET_ERROR;
  }

  for (size_t i = 0; i < input_shape.size(); ++i) {
    param_->shape_[i] = input_shape[i];
  }

  if (param_->param_length_ < DIMENSION_4D) {
    for (int i = param_->param_length_ - 1, j = 1; i >= 0; --i, ++j) {
      param_->begin_[DIMENSION_4D - j] = param_->begin_[i];
      param_->size_[DIMENSION_4D - j] = param_->size_[i];
    }
    for (size_t i = 0; i < DIMENSION_4D - param_->param_length_; i++) {
      param_->begin_[i] = 0;
      param_->size_[i] = 1;
    }
  }
  param_->param_length_ = DIMENSION_4D;
  for (int i = 0; i < DIMENSION_4D; ++i) {
    if (param_->size_[i] < 0) {
      param_->size_[i] = param_->shape_[i] - param_->begin_[i];
    }
    param_->end_[i] = param_->begin_[i] + param_->size_[i];
  }

  return RET_OK;
}

kernel::LiteKernel *CpuSliceInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Slice);
  auto *kernel = new (std::nothrow) SliceInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SliceInt8CPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuSliceFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Slice);
  auto *kernel = new (std::nothrow) SliceCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SliceCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Slice, CpuSliceInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Slice, CpuSliceFp32KernelCreator)
}  // namespace mindspore::kernel
