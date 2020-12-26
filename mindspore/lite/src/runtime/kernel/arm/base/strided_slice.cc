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
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/ops/populate/strided_slice_populate.h"

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
  if (op_parameter_ != nullptr) {
    free(op_parameter_);
    op_parameter_ = nullptr;
  }
  op_parameter_ = PopulateStridedSliceParameter(primitive_);
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "Malloc parameter failed";
    return RET_ERROR;
  }
  param_ = reinterpret_cast<StridedSliceParameter *>(op_parameter_);
  return RET_OK;
}

int StridedSliceCPUKernel::Run() {
  auto input = in_tensors_.at(0);
  MS_ASSERT(input);
  switch (input->data_type()) {
    case kNumberTypeInt8:
      param_->data_type = kDataTypeInt8;
      break;
    case kNumberTypeFloat32:
      param_->data_type = kDataTypeFloat;
      break;
    case kNumberTypeInt32:
      param_->data_type = kDataTypeInt;
      break;
    default:
      MS_LOG(ERROR) << "Not supported data type: " << input->data_type();
      return RET_ERROR;
  }
  auto output = out_tensors_.at(0);
  MS_ASSERT(output);
  auto ret = DoStridedSlice(input->data_c(), output->data_c(), param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StridedSlice error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_StridedSlice, LiteKernelCreator<StridedSliceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_StridedSlice, LiteKernelCreator<StridedSliceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_StridedSlice, LiteKernelCreator<StridedSliceCPUKernel>)
}  // namespace mindspore::kernel
