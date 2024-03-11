/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "cpu_kernel/ms_kernel/environ/environ_create.h"
#include "cpu_kernel/ms_kernel/environ/aicpu_environ_manager.h"
#include "utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"

namespace {
const char *kEnvironCreate = "EnvironCreate";
}  // namespace
namespace aicpu {
uint32_t EnvironCreateKernel::Compute(CpuKernelContext &ctx) {
  // Parse Kernel
  auto ret = ParseKernelParam(ctx);
  CUST_KERNEL_CHECK_FALSE(ctx, ret == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID, "Parse EnvironCreate failed.");
  // Generate an unique handle.
  int64_t env_handle = EnvironMgr::GetInstance().Create(ctx);
  CUST_KERNEL_LOG_DEBUG(ctx, "Create env handle:%d", env_handle);
  auto output_data = reinterpret_cast<int64_t *>(ctx.Output(kFirstOutputIndex)->GetData());
  output_data[0] = env_handle;
  return KERNEL_STATUS_OK;
}

uint32_t EnvironCreateKernel::ParseKernelParam(CpuKernelContext &ctx) const {
  CUST_KERNEL_LOG_DEBUG(ctx, "Enter ParseKernelParam.");
  if (!EnvironMgr::GetInstance().IsScalarTensor(ctx, ctx.Output(kFirstInputIndex))) {
    CUST_KERNEL_LOG_ERROR(ctx, "The output is not scalar tensor.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kEnvironCreate, EnvironCreateKernel);
}  // namespace aicpu
