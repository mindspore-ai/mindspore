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

#include "cpu_kernel/ms_kernel/environ/environ_set.h"
#include <string>
#include <memory>
#include "securec/include/securec.h"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/ms_kernel/environ/aicpu_environ_manager.h"

namespace {
constexpr auto kEnvValueTypeAttr = "value_type";
const char *kEnvironSet = "EnvironSet";
}  // namespace
namespace aicpu {
uint32_t EnvironSetKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_LOG_DEBUG(ctx, "Enter DoCompute.");
  // Parse Kernel
  auto ret = ParseKernelParam(ctx);
  CUST_KERNEL_CHECK_FALSE(ctx, ret == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID, "Parse EnvironSet failed.");
  auto &env_mgr = EnvironMgr::GetInstance();
  auto input_handle_ptr = reinterpret_cast<int64_t *>(ctx.Input(kFirstInputIndex)->GetData());
  auto input_key_ptr = reinterpret_cast<int64_t *>(ctx.Input(kSecondInputIndex)->GetData());
  auto input_value_ptr = reinterpret_cast<void *>(ctx.Input(kThirdInputIndex)->GetData());
  auto output_handle_ptr = reinterpret_cast<int64_t *>(ctx.Output(kFirstOutputIndex)->GetData());

  auto *value_ptr = malloc(value_size_);
  CUST_KERNEL_CHECK_NULLPTR(ctx, value_ptr, KERNEL_STATUS_INNER_ERROR, "Malloc failed.")
  auto res = memcpy_s(value_ptr, value_size_, input_value_ptr, value_size_);
  CUST_KERNEL_CHECK_FALSE(ctx, (res == EOK), KERNEL_STATUS_INNER_ERROR, "Memcpy size from input[2] to environ failed.",
                          value_size_);

  // Set env member.
  const auto &env = env_mgr.Get(input_handle_ptr[0]);
  CUST_KERNEL_CHECK_FALSE(ctx, env, KERNEL_STATUS_INNER_ERROR, "Get handle[%d] failed.", input_handle_ptr[0]);

  auto env_value = std::make_shared<EnvironValue>(value_ptr, value_size_, attr_value_type_);
  env->Set(input_key_ptr[0], env_value);
  CUST_KERNEL_LOG_DEBUG(ctx, "EnvironSetKernel: handle[%d], key[%d], value[%d]", input_handle_ptr[0], input_key_ptr[0],
                        (void *)&env_value);

  // Set output handle
  output_handle_ptr[0] = input_handle_ptr[0];
  return KERNEL_STATUS_OK;
}

uint32_t EnvironSetKernel::ParseKernelParam(CpuKernelContext &ctx) {
  CUST_KERNEL_LOG_DEBUG(ctx, "Enter ParseKernelParam.");
  auto &env_mgr = EnvironMgr::GetInstance();
  if (!env_mgr.CheckEnvInput(ctx)) {
    CUST_KERNEL_LOG_DEBUG(ctx, "The input checks invalid. ");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (!env_mgr.IsScalarTensor(ctx, ctx.Output(kFirstOutputIndex))) {
    CUST_KERNEL_LOG_ERROR(ctx, "The output handle is not equal of input handle.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // Get value type.
  auto value_type_ptr = ctx.GetAttr(kEnvValueTypeAttr);
  CUST_KERNEL_CHECK_NULLPTR(ctx, value_type_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr value_type failed.");
  attr_value_type_ = value_type_ptr->GetInt();

  // Get value size.
  auto *value_tensor = ctx.Input(kThirdInputIndex);
  value_size_ = value_tensor->GetDataSize();
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kEnvironSet, EnvironSetKernel);
}  // namespace aicpu
