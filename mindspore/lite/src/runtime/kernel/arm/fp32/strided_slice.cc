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

#include "src/runtime/kernel/arm/fp32/strided_slice.h"
#include <vector>
#include "src/runtime/kernel/arm/opclib/fp32/strided_slice.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_StridedSlice;

namespace mindspore::kernel {

int StridedSliceCPUKernel::Init() { return RET_OK; }

int StridedSliceCPUKernel::ReSize() { return 0; }

int StridedSliceCPUKernel::StridedSlice() {
  StridedSliceParameter *param = reinterpret_cast<StridedSliceParameter *>(opParameter);
  auto ret = DoStridedSlice(input_ptr_, output_ptr_, param);
  if (ret != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceCPUKernel::Run() {
  auto input_tensor = inputs_.at(0);
  auto output_tensor = outputs_.at(0);
  input_ptr_ = reinterpret_cast<float *>(input_tensor->Data());
  output_ptr_ = reinterpret_cast<float *>(output_tensor->Data());

  auto ret = StridedSlice();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StridedSlice error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuStridedSliceFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                     const std::vector<lite::tensor::Tensor *> &outputs,
                                                     OpParameter *opParameter, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc) {
  MS_ASSERT(desc.type == schema::PrimitiveType_StridedSlice);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "opParameter null pointer dereferencing.";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) StridedSliceCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != 0) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_StridedSlice, CpuStridedSliceFp32KernelCreator)
}  // namespace mindspore::kernel
