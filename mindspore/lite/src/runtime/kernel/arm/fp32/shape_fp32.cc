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

#include "src/runtime/kernel/arm/fp32/shape_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Shape;

namespace mindspore::kernel {
int ShapeCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  return RET_OK;
}

int ShapeCPUKernel::ReSize() { return RET_OK; }

int ShapeCPUKernel::Run() {
  auto out_tensor = out_tensors_.front();
  auto in_tensor = in_tensors_.front();
  if (in_tensor == nullptr || out_tensor == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_ERROR;
  }
  if (in_tensor->MutableData() == nullptr || out_tensor->MutableData() == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < in_tensor->shape().size(); i++) {
    reinterpret_cast<int *>(out_tensor->MutableData())[i] = in_tensor->shape().at(i);
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Shape, LiteKernelCreator<ShapeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Shape, LiteKernelCreator<ShapeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Shape, LiteKernelCreator<ShapeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Shape, LiteKernelCreator<ShapeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimitiveType_Shape, LiteKernelCreator<ShapeCPUKernel>)
}  // namespace mindspore::kernel
