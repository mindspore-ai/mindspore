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

#include "src/runtime/kernel/arm/fp16/fullconnection_fp16.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
void FullconnectionFP16CPUKernel::InitAShape() {
  auto a_shape = in_tensors_.at(0)->shape();
  params_->row_ = a_shape[0];
  params_->deep_ = a_shape[1];
}

void FullconnectionFP16CPUKernel::InitBShape() {
  auto b_shape = in_tensors_.at(1)->shape();
  params_->col_ = b_shape[0];
  params_->deep_ = b_shape[1];
}

int FullconnectionFP16CPUKernel::ReSize() {
  InitAShape();
  InitBShape();
  return MatmulBaseFP16CPUKernel::ReSize();
}

int FullconnectionFP16CPUKernel::Init() {
  params_->batch = 1;
  params_->a_transpose_ = false;
  params_->b_transpose_ = true;

  MatmulBaseFP16CPUKernel::InitParameter();

  if (params_->a_const_ == true) {
    InitAShape();
  }

  if (params_->b_const_ == true) {
    InitBShape();
  }

  auto ret = MatmulBaseFP16CPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int FullconnectionFP16CPUKernel::Run() {
  auto ret = MatmulBaseFP16CPUKernel::Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FullconnectionFP16CPUKernel run failed";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FullConnection, LiteKernelCreator<FullconnectionFP16CPUKernel>)
}  // namespace mindspore::kernel
