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
#include <string>
#include <functional>
#include <vector>
#include "transform/acl_ir/op_api_exec.h"
#include "runtime/device/device_address_utils.h"

#define LAUNCH_ACLNN(aclnn_api, device_context, stream_id, ...)                                                    \
  do {                                                                                                             \
    static const std::string aclnn_name = #aclnn_api;                                                              \
    runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,                                   \
                                             runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false);      \
    auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);                                   \
    auto [ws_size, executor_handle, release_function] = GEN_EXECUTOR(aclnn_name, __VA_ARGS__);                     \
    if (ws_size == 0) {                                                                                            \
      DispatchLaunchKernel(device_context, aclnn_name, nullptr, 0, executor_handle, stream_ptr, release_function); \
    } else {                                                                                                       \
      auto workspace_device_address =                                                                              \
        runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context, stream_id, ws_size);                   \
      DispatchLaunchKernel(device_context, aclnn_name, workspace_device_address->GetMutablePtr(), ws_size,         \
                           executor_handle, stream_ptr, release_function);                                         \
    }                                                                                                              \
    static auto sync = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);              \
    if (sync) {                                                                                                    \
      if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {                                      \
        MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;                                            \
      }                                                                                                            \
    } else {                                                                                                       \
      runtime::DeviceAddressUtils::ProcessCrossStreamAddress(aclnn_name, device_context, stream_id, __VA_ARGS__);  \
    }                                                                                                              \
  } while (false)

#define LAUNCH_ACLNN_SYNC(aclnn_api, device_context, stream_id, ...)                                             \
  [](const std::string &aclnn_name, const device::DeviceContext *device_context, size_t real_stream_id,          \
     auto &... args) -> auto {                                                                                   \
    runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,                                 \
                                             runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false);    \
    auto stream_ptr = device_context->device_res_manager_->GetStream(real_stream_id);                            \
    auto use_huge_pages = true;                                                                                  \
    auto return_values = GEN_EXECUTOR_CUST(aclnn_name, use_huge_pages, args...);                                 \
    auto ws_size = std::get<0>(return_values);                                                                   \
    auto executor_handle = std::get<1>(return_values);                                                           \
    if (ws_size == 0) {                                                                                          \
      LaunchKernel(aclnn_name, nullptr, 0, executor_handle, stream_ptr);                                         \
    } else {                                                                                                     \
      auto workspace_device_address =                                                                            \
        runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context, real_stream_id, ws_size);            \
      LaunchKernel(aclnn_name, workspace_device_address->GetMutablePtr(), ws_size, executor_handle, stream_ptr); \
    }                                                                                                            \
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {                                      \
      MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;                                            \
    }                                                                                                            \
    return return_values;                                                                                        \
  }                                                                                                              \
  (#aclnn_api, device_context, stream_id, __VA_ARGS__)
namespace mindspore {
namespace kernel {
namespace pyboost {
int8_t GetCubeMathType();
void DispatchLaunchKernel(const DeviceContext *device_context, const std::string &aclnn_name, void *ws_ptr,
                          size_t ws_size, transform::aclOpExecutor *executor, void *stream,
                          const std::function<void()> &release_func);
void LaunchKernel(const std::string &aclnn_name, void *ws_ptr, size_t ws_size, transform::aclOpExecutor *executor,
                  void *stream);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
