/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/fullconnection_fp32.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
int FullconnectionCPUKernel::Prepare() {
  CHECK_NULL_RETURN(matmul_base_);
  matmul_base_->set_name(name_);
  matmul_base_->set_workspace(workspace());
  return matmul_base_->FullConnectionPrepare();
}

int FullconnectionCPUKernel::ReSize() {
  CHECK_NULL_RETURN(matmul_base_);
  matmul_base_->set_workspace(workspace());
  return matmul_base_->FullConnectionReSize();
}

int FullconnectionCPUKernel::Run() {
  CHECK_NULL_RETURN(matmul_base_);
  matmul_base_->set_workspace(workspace());
  return matmul_base_->Run();
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FullConnection, LiteKernelCreator<FullconnectionCPUKernel>)
}  // namespace mindspore::kernel
