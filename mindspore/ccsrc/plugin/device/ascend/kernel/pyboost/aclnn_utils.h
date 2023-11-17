/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
#include <algorithm>
#include <functional>
#include "transform/acl_ir/op_api_exec.h"
#include "runtime/device/device_address_utils.h"

#define LAUNCH_ACLNN(aclnn_api, device_context, stream_ptr, ...)                                                  \
  static const std::string aclnn_name = #aclnn_api;                                                               \
  auto [ws_size, executor_handle, release_function] = GEN_EXECUTOR(aclnn_name, __VA_ARGS__);                      \
  if (ws_size == 0) {                                                                                             \
    RUN_OP_API_ASYNC(aclnn_name, nullptr, 0, executor_handle, stream_ptr, release_function);                      \
  } else {                                                                                                        \
    auto workspace_device_address = runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context, ws_size); \
    RUN_OP_API_ASYNC(aclnn_name, workspace_device_address->GetMutablePtr(), ws_size, executor_handle, stream_ptr, \
                     release_function);                                                                           \
  }

namespace mindspore {
namespace kernel {
namespace pyboost {
template <typename T>
std::vector<T> ConvertValueTupleToVector(const ValueTuplePtr &tuple) {
  std::vector<T> result;
  const auto &values = tuple->value();
  for (const auto &value : values) {
    (void)result.emplace_back(GetValue<T>(value));
  }
  MS_LOG(DEBUG) << "Convert ValueTuple to vector " << result;
  return result;
}
int8_t GetCubeMathType();
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
