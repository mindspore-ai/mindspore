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

#include "src/runtime/kernel/arm/base/strided_slice.h"
#include <vector>
#include "src/runtime/kernel/arm/nnacl/strided_slice.h"
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

int StridedSliceCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int StridedSliceCPUKernel::ReSize() {
  auto input = in_tensors_.at(0);
  auto parameter = reinterpret_cast<StridedSliceParameter *>(op_parameter_);
  MS_ASSERT(input);
  MS_ASSERT(parameter);
  parameter->data_type = input->data_type() == kNumberTypeInt8 ? kDataTypeInt8 : kDataTypeFloat;
  return RET_OK;
}

int StridedSliceCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }

  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);

  ret = DoStridedSlice(input->Data(), output->Data(), reinterpret_cast<StridedSliceParameter *>(op_parameter_));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StridedSlice error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuStridedSliceKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                 const std::vector<lite::tensor::Tensor *> &outputs,
                                                 OpParameter *opParameter, const lite::Context *ctx,
                                                 const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_StridedSlice);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "opParameter null pointer dereferencing.";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) StridedSliceCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_StridedSlice, CpuStridedSliceKernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_StridedSlice, CpuStridedSliceKernelCreator)
}  // namespace mindspore::kernel
