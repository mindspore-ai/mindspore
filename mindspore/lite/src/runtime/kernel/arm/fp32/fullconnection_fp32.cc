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

#include "src/runtime/kernel/arm/fp32/fullconnection_fp32.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
int FullconnectionCPUKernel::Init() {
  MatmulFp32BaseCPUKernel::InitParameter();

  if (params_->a_const_ == true) {
    auto a_shape = in_tensors_.at(0)->shape();
    params_->row_ = a_shape[0];
    params_->deep_ = a_shape[1];
  }

  if (params_->b_const_ == true) {
    auto b_shape = in_tensors_.at(1)->shape();
    params_->col_ = b_shape[0];
    params_->deep_ = b_shape[1];
  }

  params_->batch = 1;
  params_->a_transpose_ = false;
  params_->b_transpose_ = true;

  auto ret = MatmulFp32BaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int FullconnectionCPUKernel::ReSize() {
  int row = 1;
  for (size_t i = 0; i < out_tensors_.at(0)->shape().size() - 1; ++i) {
    row *= (out_tensors_.at(0)->shape())[i];
  }
  params_->row_ = row;
  params_->col_ = out_tensors_.at(0)->shape().back();
  params_->deep_ = (in_tensors_.at(1)->shape()).at(1);

  return MatmulFp32BaseCPUKernel::ReSize();
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FullConnection, LiteKernelCreator<FullconnectionCPUKernel>)
}  // namespace mindspore::kernel
