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

#include "src/litert/kernel/cpu/base/assert.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Assert;

namespace mindspore::kernel {
int AssertCPUKernel::Prepare() {
  CHECK_NOT_EQUAL_RETURN(in_tensors_.size(), 1);
  CHECK_NOT_EQUAL_RETURN(out_tensors_.size(), 1);
  return RET_OK;
}

int AssertCPUKernel::ReSize() { return RET_OK; }

int AssertCPUKernel::Run() {
  CHECK_NULL_RETURN(in_tensors_.front());
  CHECK_NULL_RETURN(in_tensors_.front()->data());
  auto cond = reinterpret_cast<bool *>(in_tensors_.front()->data());
  if (*cond) {
    return RET_OK;
  } else {
    for (size_t i = 1; i < in_tensors_.size(); i++) {
      CHECK_NULL_RETURN(in_tensors_.at(i));
      MS_LOG(ERROR) << in_tensors_.at(i)->ToString();
    }
    return RET_ERROR;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Assert, LiteKernelCreator<AssertCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Assert, LiteKernelCreator<AssertCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Assert, LiteKernelCreator<AssertCPUKernel>)
}  // namespace mindspore::kernel
