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

#include "src/runtime/kernel/arm/int8/fullconnection_int8.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
int FullconnectionInt8CPUKernel::Init() {
  param_->batch = 1;
  param_->a_transpose_ = false;
  param_->b_transpose_ = true;

  InitParameter();

  auto ret = MatmulBaseInt8CPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParallelLaunch failed";
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int FullconnectionInt8CPUKernel::ReSize() {
  int row = 1;
  for (size_t i = 0; i < out_tensors_.at(0)->shape().size() - 1; ++i) {
    row *= (out_tensors_.at(0)->shape()).at(i);
  }
  param_->row_ = row;
  param_->col_ = out_tensors_.at(0)->shape().back();
  param_->deep_ = (in_tensors_.at(1)->shape()).at(1);

  auto ret = MatmulBaseInt8CPUKernel::ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulBaseInt8CPUKernel failed";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_FullConnection, LiteKernelCreator<FullconnectionInt8CPUKernel>)
}  // namespace mindspore::kernel
