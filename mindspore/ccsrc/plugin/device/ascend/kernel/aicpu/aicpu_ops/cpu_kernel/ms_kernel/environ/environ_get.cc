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

#include "cpu_kernel/ms_kernel/environ/environ_get.h"
#include <random>
#include <climits>
#include <vector>
#include <algorithm>
#include <string>
#include "securec/include/securec.h"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/ms_kernel/environ/aicpu_environ_manager.h"

namespace {
constexpr auto kEnvValueTypeAttr = "value_type";
const char *kEnvironGet = "EnvironGet";
}  // namespace

namespace aicpu {
uint32_t EnvironGetKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_LOG_DEBUG(ctx, "Enter DoCompute.");
  // Parse Kernel
  auto ret = ParseKernelParam(ctx);
  CUST_KERNEL_CHECK_FALSE(ctx, ret == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID, "Parse EnvironGet failed.");
  auto &env_mgr = EnvironMgr::GetInstance();
  auto input_handle_ptr = reinterpret_cast<int64_t *>((ctx.Input(kFirstInputIndex)->GetData()));
  auto input_key_ptr = reinterpret_cast<int64_t *>((ctx.Input(kSecondInputIndex)->GetData()));
  auto default_value_ptr = reinterpret_cast<void *>((ctx.Input(kThirdInputIndex)->GetData()));
  auto output_ptr = reinterpret_cast<void *>(ctx.Output(kFirstOutputIndex)->GetData());

  // Get handle and key
  int64_t handle = input_handle_ptr[0];
  int64_t key = input_key_ptr[0];

  // Get env and value by handle and key
  const auto &env = env_mgr.Get(handle);
  CUST_KERNEL_CHECK_NULLPTR(ctx, env, KERNEL_STATUS_PARAM_INVALID, "Get env [%d] failed", handle)
  const auto &env_value = env->Get(key);

  CUST_KERNEL_LOG_DEBUG(ctx, "EnvironGetKernel: hindle[%d], key[%d], value[%d]", handle, key, (void *)&env_value);
  // Default value
  auto output_value_ptr = default_value_ptr;
  auto output_value_size = default_value_size_;
  auto output_value_type = attr_value_type_;
  if (env_value != nullptr) {
    output_value_ptr = env_value->addr_;
    output_value_size = env_value->size_;
    output_value_type = env_value->value_type_;
  } else {
    CUST_KERNEL_LOG_ERROR(ctx, "Get key[%d] value checks failed.", key);
  }

  if ((output_value_size_ < output_value_size) || (output_value_type != attr_value_type_)) {
    CUST_KERNEL_LOG_ERROR(ctx, "The env value checks invalid, value_size: %d vs %d, value_type:%d vs %d",
                          output_value_size_, output_value_size, output_value_type, attr_value_type_);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto res = memcpy_s(output_ptr, output_value_size_, output_value_ptr, output_value_size_);
  CUST_KERNEL_CHECK_FALSE(ctx, (res == EOK), KERNEL_STATUS_PARAM_INVALID,
                          "Memcpy size[%zu] from env map to output[0] failed.", output_value_size_);

  return KERNEL_STATUS_OK;
}

uint32_t EnvironGetKernel::ParseKernelParam(CpuKernelContext &ctx) {
  CUST_KERNEL_LOG_DEBUG(ctx, "Enter ParseKernelParam.");
  auto &env_mgr = EnvironMgr::GetInstance();
  if (!env_mgr.CheckEnvInput(ctx)) {
    CUST_KERNEL_LOG_ERROR(ctx, "The input checks invalid. ");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // Get value type attr
  auto value_type_ptr = ctx.GetAttr(kEnvValueTypeAttr);
  CUST_KERNEL_CHECK_NULLPTR(ctx, value_type_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr value_type failed.");
  attr_value_type_ = value_type_ptr->GetInt();

  // check output value
  auto default_value_tensor = ctx.Input(kThirdInputIndex);
  auto output_value_ptr_tensor = ctx.Output(kFirstOutputIndex);
  CUST_KERNEL_CHECK_NULLPTR(ctx, default_value_tensor, KERNEL_STATUS_PARAM_INVALID, "Default value tensor is nullptr.");
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_value_ptr_tensor, KERNEL_STATUS_PARAM_INVALID,
                            "Output value tensor is nullptr.");
  auto default_value_shape = default_value_tensor->GetTensorShape()->GetDimSizes();
  auto output_shape = output_value_ptr_tensor->GetTensorShape()->GetDimSizes();
  auto default_value_type = default_value_tensor->GetDataType();
  auto output_type = output_value_ptr_tensor->GetDataType();
  if ((output_shape != default_value_shape) || (output_type != default_value_type)) {
    CUST_KERNEL_LOG_ERROR(ctx, "The env value checks invalid.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // Get value size.
  default_value_size_ = default_value_tensor->GetDataSize();
  output_value_size_ = output_value_ptr_tensor->GetDataSize();

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kEnvironGet, EnvironGetKernel);
}  // namespace aicpu
