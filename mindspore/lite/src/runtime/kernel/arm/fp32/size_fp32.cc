/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/size_fp32.h"
#include "src/kernel_registry.h"
#include "schema/model_generated.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Size;

namespace mindspore::kernel {
int SizeCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  return RET_OK;
}

int SizeCPUKernel::ReSize() { return RET_OK; }

int SizeCPUKernel::Run() {
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  CHECK_NULL_RETURN(in_tensor->data());
  CHECK_NULL_RETURN(out_tensor->data());
  reinterpret_cast<int *>(out_tensor->data())[0] = in_tensor->ElementsNum();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Size, LiteKernelCreator<SizeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Size, LiteKernelCreator<SizeCPUKernel>)
}  // namespace mindspore::kernel
