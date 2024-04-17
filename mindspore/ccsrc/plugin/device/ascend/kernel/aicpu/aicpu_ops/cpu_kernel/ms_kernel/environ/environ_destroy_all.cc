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

#include "cpu_kernel/ms_kernel/environ/environ_destroy_all.h"
#include "cpu_kernel/ms_kernel/environ/aicpu_environ_manager.h"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const char *kEnvironDestroyAll = "EnvironDestroyAll";
}  // namespace
namespace aicpu {
uint32_t EnvironDestroyAllKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_LOG_DEBUG(ctx, "Destroy all env handle");
  // Parse Kernel
  auto ret = ParseKernelParam(ctx);
  CUST_KERNEL_CHECK_FALSE(ctx, ret == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID, "Parse EnvironDestroyAll failed.");
  EnvironMgr::GetInstance().Clear(ctx);
  return KERNEL_STATUS_OK;
}

uint32_t EnvironDestroyAllKernel::ParseKernelParam(CpuKernelContext &ctx) const {
  CUST_KERNEL_LOG_DEBUG(ctx, "Enter ParseKernelParam.");
  if (!EnvironMgr::GetInstance().IsScalarTensor(ctx, ctx.Output(kFirstInputIndex))) {
    CUST_KERNEL_LOG_ERROR(ctx, "The output is not scalar tensor.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kEnvironDestroyAll, EnvironDestroyAllKernel);
}  // namespace aicpu
