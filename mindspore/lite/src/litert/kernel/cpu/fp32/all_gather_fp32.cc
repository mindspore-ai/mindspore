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

#include "src/litert/kernel/cpu/fp32/all_gather_fp32.h"
#include "schema/ops_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_AllGather;

namespace mindspore::kernel {
int AllGatherCPUKernel::Prepare() {
  MS_LOG(ERROR) << "unsupported AllGather kernel";
  return lite::RET_NOT_SUPPORT;
}

int AllGatherCPUKernel::ReSize() { return lite::RET_OK; }

int AllGatherCPUKernel::Run() {
  int rank = param_->rank_size_;
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_NULL_RETURN(out_tensors_.front());
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(out_tensors_.front());
  size_t data_size = in_tensors().front()->Size();
  auto out_tensor = out_tensors().front();
  int8_t *out_data = reinterpret_cast<int8_t *>(out_tensor->data());
  CHECK_NULL_RETURN(out_data);

  for (int i = 0; i < rank; i++) {
    /* update in_tensor by rank id */
    auto in_tensor = in_tensors().front();
    memcpy(out_data + i * data_size, in_tensor->data(), data_size);
  }

  return lite::RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AllGather, LiteKernelCreator<AllGatherCPUKernel>)
}  // namespace mindspore::kernel
