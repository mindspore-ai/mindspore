/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_PYBOOST_ACLNN_UTILS_H_
#define MINDSPORE_CCSRC_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_PYBOOST_ACLNN_UTILS_H_

#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/runtime/task/device_task.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"

/**
 * @brief Macro to dispatch and launch ACLNN operations asynchronously using PyBoostDeviceTask
 *
 * This macro wraps the common pattern of creating a PyBoostDeviceTask and launching
 * an ACLNN operation. It uses an IIFE pattern to handle parameter capture properly.
 *
 * @param aclnn_api The ACLNN API function name (e.g., aclnnAdd, aclnnMatmul)
 * @param ... Variable arguments to pass to LAUNCH_ACLNN (excluding device_context and stream_id)
 *
 * @note Requires 'op' variable (OpRunner pointer) to be available in current scope
 * @note Parameter order must match LAUNCH_ACLNN's expected signature
 * @note All variadic arguments are captured by value in the outer lambda
 *
 * @warning CRITICAL: Always pass nullptr directly instead of typed null pointers!
 *         NEVER use: TensorPtr null_var = nullptr; DISPATCH_LAUNCH_ACLNN(..., null_var, ...)
 *         Always use: DISPATCH_LAUNCH_ACLNN(..., nullptr, ...)
 *         This is because FillAddress uses compile-time type checking (std::is_same_v<T, std::nullptr_t>)
 *         and typed null pointers will cause precision issues.
 */
#define DISPATCH_LAUNCH_ACLNN(aclnn_api, ...)                                           \
  do {                                                                                  \
    auto __device_context = op->device_context();                                       \
    auto __stream_id = op->stream_id();                                                 \
    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(             \
      [=]() { LAUNCH_ACLNN(aclnn_api, __device_context, __stream_id, __VA_ARGS__); })); \
  } while (false)

// Variant that uses CurrentDeviceContext/CurrentStream instead of OpRunner
#define DISPATCH_ACLNN(aclnn_api, ...)                                                  \
  do {                                                                                  \
    auto __device_context = CurrentDeviceContext::get();                                \
    auto __stream_id = CurrentStream::id();                                             \
    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(             \
      [=]() { LAUNCH_ACLNN(aclnn_api, __device_context, __stream_id, __VA_ARGS__); })); \
  } while (false)

#endif  // MINDSPORE_CCSRC_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_PYBOOST_ACLNN_UTILS_H_
