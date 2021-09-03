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
#include "src/runtime/kernel/arm/fp16/depth_to_space_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DepthToSpace;

namespace mindspore::kernel {
int DepthToSpaceFp16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  param_->data_type_size_ = sizeof(float16_t);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_DepthToSpace, LiteKernelCreator<DepthToSpaceFp16CPUKernel>)
}  // namespace mindspore::kernel
