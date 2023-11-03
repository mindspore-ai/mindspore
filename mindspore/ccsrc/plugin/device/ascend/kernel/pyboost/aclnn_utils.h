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

// TODO: Get kCubeMathType by SOC version
#define LAUNCH_ACLNN_CUBE(aclnn_api, stream_ptr, ...)                                                        \
  int8_t cube_math_type = 0;                                                                                 \
  auto [workspace_size, executor, after_launch_func] = GEN_EXECUTOR(aclnn_api, __VA_ARGS__, cube_math_type); \
  if (workspace_size == 0) {                                                                                 \
    RUN_OP_API(aclnn_api, stream_ptr, nullptr, 0, executor, after_launch_func);                              \
  } else {                                                                                                   \
    auto workspace_device_address =                                                                          \
      runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context_, workspace_size);                  \
    RUN_OP_API(aclnn_api, stream_ptr, workspace_device_address->GetMutablePtr(), workspace_size, executor,   \
               after_launch_func);                                                                           \
  }

#define LAUNCH_ACLNN(aclnn_api, stream_ptr, ...)                                                           \
  auto [workspace_size, executor, after_launch_func] = GEN_EXECUTOR(aclnn_api, __VA_ARGS__);               \
  if (workspace_size == 0) {                                                                               \
    RUN_OP_API(aclnn_api, stream_ptr, nullptr, 0, executor, after_launch_func);                            \
  } else {                                                                                                 \
    auto workspace_device_address =                                                                        \
      runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context_, workspace_size);                \
    RUN_OP_API(aclnn_api, stream_ptr, workspace_device_address->GetMutablePtr(), workspace_size, executor, \
               after_launch_func);                                                                         \
  }

namespace mindspore {
namespace kernel {
namespace pyboost {
template <typename T>
std::vector<T> ConvertValueTupleToVector(const ValueTuplePtr &tuple);
int8_t GetCubeMathType();
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
