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
#include "src/runtime/kernel/arm/fp32/slice.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/opclib/fp32/slice.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Slice;

namespace mindspore::kernel {

int SliceCPUKernel::Init() {
  auto *param = reinterpret_cast<SliceParameter *>(opParameter);
  auto input_shape = inputs_[0]->shape();
  if (input_shape.size() != param->param_length_) {
    MS_LOG(ERROR) << "Input begin's lenth " << param->param_length_ << "is not equal to input shape size "
                  << input_shape.size();
    return RET_ERROR;
  }
  if (input_shape.size() > SLICE_SHAPE_MAX_SIZE) {
    MS_LOG(ERROR) << "input dimension num should <= " << SLICE_SHAPE_MAX_SIZE;
    return RET_ERROR;
  }

  for (size_t i = 0; i < input_shape.size(); ++i) {
    param->shape_[i] = input_shape[i];
  }
  return RET_OK;
}

int SliceCPUKernel::Run() {
  SliceParameter *param = reinterpret_cast<SliceParameter *>(opParameter);
  const float *input_data = reinterpret_cast<const float *>(inputs_[0]->Data());
  float *output_data = reinterpret_cast<float *>(outputs_[0]->Data());

  return DoSlice(input_data, param, output_data);
}

kernel::LiteKernel *CpuSliceFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) SliceCPUKernel(opParameter, inputs, outputs);
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

REG_KERNEL(kCPU, PrimitiveType_Slice, CpuSliceFp32KernelCreator)
}  // namespace mindspore::kernel

