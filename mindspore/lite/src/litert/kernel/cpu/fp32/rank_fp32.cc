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

#include "src/litert/kernel/cpu/fp32/rank_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Rank;

namespace mindspore::kernel {
int RankCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[kInputIndex]);
  CHECK_NULL_RETURN(out_tensors_[kOutputIndex]);
  return RET_OK;
}

int RankCPUKernel::ReSize() { return RET_OK; }

int RankCPUKernel::Run() {
  auto output_data = out_tensors_[kOutputIndex]->data();
  CHECK_NULL_RETURN(output_data);
  size_t rank = in_tensors_[kInputIndex]->shape().size();
  if (in_tensors_[kInputIndex]->data_type() == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    *static_cast<float16_t *>(output_data) = static_cast<float16_t>(rank);
#endif
  } else {
    *static_cast<float *>(output_data) = static_cast<float>(rank);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Rank, LiteKernelCreator<RankCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Rank, LiteKernelCreator<RankCPUKernel>)
#endif
}  // namespace mindspore::kernel
