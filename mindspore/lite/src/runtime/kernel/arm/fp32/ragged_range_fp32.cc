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
#include "src/runtime/kernel/arm/fp32/ragged_range_fp32.h"
#include <vector>
#include "nnacl/fp32/ragged_range_fp32.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_RaggedRange;

namespace mindspore::kernel {
int RaggedRangeCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 3);
  CHECK_LESS_RETURN(out_tensors_.size(), 2);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int RaggedRangeCPUKernel::ReSize() { return RET_OK; }

int RaggedRangeCPUKernel::Run() {
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    RaggedRangeFp32(static_cast<float *>(in_tensors_.at(0)->data()), static_cast<float *>(in_tensors_.at(1)->data()),
                    static_cast<float *>(in_tensors_.at(2)->data()), static_cast<int *>(out_tensors_.at(0)->data()),
                    static_cast<float *>(out_tensors_.at(1)->data()),
                    reinterpret_cast<RaggedRangeParameter *>(op_parameter_));
  } else {
    RaggedRangeInt(static_cast<int *>(in_tensors_.at(0)->data()), static_cast<int *>(in_tensors_.at(1)->data()),
                   static_cast<int *>(in_tensors_.at(2)->data()), static_cast<int *>(out_tensors_.at(0)->data()),
                   static_cast<int *>(out_tensors_.at(1)->data()),
                   reinterpret_cast<RaggedRangeParameter *>(op_parameter_));
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_RaggedRange, LiteKernelCreator<RaggedRangeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_RaggedRange, LiteKernelCreator<RaggedRangeCPUKernel>)
}  // namespace mindspore::kernel
