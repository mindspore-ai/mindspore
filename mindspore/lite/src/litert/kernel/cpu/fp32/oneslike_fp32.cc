/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/oneslike_fp32.h"
#include "schema/model_generated.h"
#include "nnacl/base/zeroslike_base.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_OnesLike;

namespace mindspore::kernel {
int OnesLikeCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  return RET_OK;
}

int OnesLikeCPUKernel::Run() {
  auto output = out_tensors_[0];
  CHECK_NULL_RETURN(output);
  if (output->data_type() == kNumberTypeInt32) {
    ApproximateOnesLike(static_cast<int *>(output->data()), output->ElementsNum());
  } else if (output->data_type() == kNumberTypeFloat32) {
    ApproximateOnesLike(static_cast<float *>(output->data()), output->ElementsNum());
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_OnesLike, LiteKernelCreator<OnesLikeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_OnesLike, LiteKernelCreator<OnesLikeCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_OnesLike, LiteKernelCreator<OnesLikeCPUKernel>)
#endif
}  // namespace mindspore::kernel
