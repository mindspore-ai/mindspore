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

#include "cpu_kernel/ms_kernel/environ/aicpu_environ_manager.h"
#include <string>
#include "utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"

namespace aicpu {
constexpr auto kScalarTensorShapeDim = 1;
constexpr auto kScalarTensorShapeSize = 1;
constexpr auto kEnvValueTypeAttr = "value_type";

int64_t EnvironMgr::Create(CpuKernelContext &ctx) {
  std::unique_lock<std::mutex> lock(mutex);
  if (env_handles_count_ >= INT64_MAX) {
    CUST_KERNEL_LOG_ERROR(ctx, " The handles number:%d is out of range: ", env_handles_count_);
    return KERNEL_STATUS_INNER_ERROR;
  }
  int64_t ret_handle = ++env_handles_count_;
  auto env = std::make_shared<Environ>(ret_handle);
  CUST_KERNEL_CHECK_NULLPTR(ctx, env, KERNEL_STATUS_INNER_ERROR, "env is null.");
  envs_[ret_handle] = env;

  return ret_handle;
}

EnvironPtr EnvironMgr::Get(int64_t handle) {
  std::unique_lock<std::mutex> lock(mutex);
  const auto &envIter = envs_.find(handle);
  if (envIter != envs_.end()) {
    auto &result = envIter->second;
    return result;
  }
  return nullptr;
}

void EnvironMgr::Clear(CpuKernelContext &ctx) {
  std::unique_lock<std::mutex> lock(mutex);
  for (auto &env : envs_) {
    CUST_KERNEL_CHECK_NULLPTR_VOID(ctx, env.second, "env is null.")
    env.second->Clear(ctx);
  }
  envs_.clear();
}

bool EnvironMgr::IsScalarTensor(CpuKernelContext &ctx, const Tensor *tensor) const {
  CUST_KERNEL_CHECK_NULLPTR(ctx, tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "tensor is nullptr.");
  auto shape_ptr = tensor->GetTensorShape();
  CUST_KERNEL_CHECK_NULLPTR(ctx, shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get tensor shape failed.");
  auto shape = shape_ptr->GetDimSizes();
  if (shape.empty()) {
    CUST_KERNEL_LOG_DEBUG(ctx, "The shape is empty.");
    return true;
  }

  if ((shape.size() == kScalarTensorShapeDim) && (shape[0] == kScalarTensorShapeSize)) {
    CUST_KERNEL_LOG_DEBUG(ctx, "The tensor is scalar.");
    return true;
  }
  return false;
}

bool EnvironMgr::CheckEnvInput(CpuKernelContext &ctx) const {
  auto *value_type_ptr = ctx.GetAttr(kEnvValueTypeAttr);
  CUST_KERNEL_CHECK_NULLPTR(ctx, value_type_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr value_type failed.");
  auto value_type_attr = value_type_ptr->GetInt();
  if ((value_type_attr != kObjectTypeTensorType) && (value_type_attr != kObjectTypeEnvType)) {
    CUST_KERNEL_LOG_ERROR(ctx, "The value type is not supported: [%d]", value_type_attr);
    return false;
  }

  // Check the input handle.
  if (!IsScalarTensor(ctx, ctx.Input(kFirstInputIndex))) {
    CUST_KERNEL_LOG_ERROR(ctx, "The input handle checks invalid.");
    return false;
  }

  // Check the input key
  if (!IsScalarTensor(ctx, ctx.Input(kSecondInputIndex))) {
    CUST_KERNEL_LOG_ERROR(ctx, "The input key checks invalid.");
    return false;
  }

  // Check the input value
  if ((value_type_attr == kObjectTypeEnvType) && (!IsScalarTensor(ctx, ctx.Input(kThirdInputIndex)))) {
    CUST_KERNEL_LOG_ERROR(ctx, "The input value checks invalid.");
    return false;
  }

  return true;
}
}  // namespace aicpu
