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
#include <string>
#include <vector>
#include <utility>
#include "runtime/device/device_address_utils.h"
#include "runtime/pipeline/pipeline.h"
#include "transform/acl_ir/op_api_exec.h"

#define DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, ws_ptr, ws_size, executor, stream, release_func) \
  runtime::OpExecutor::DispatchLaunchTask([=]() {                                                           \
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,                                  \
                                       runtime::ProfilerEvent::kPyNativeLaunchTask, aclnn_name, false);     \
    MS_LOG(DEBUG) << "launch task start, " << aclnn_name;                                                   \
    device_context->device_res_manager_->BindDeviceToCurrentThread(false);                                  \
    RUN_OP_API_ASYNC(aclnn_name, ws_ptr, ws_size, executor, stream, release_func);                          \
    MS_LOG(DEBUG) << "launch task end, " << aclnn_name;                                                     \
  });

#define LAUNCH_ACLNN(aclnn_api, device_context, stream_id, ...)                                                      \
  do {                                                                                                               \
    static const std::string aclnn_name = #aclnn_api;                                                                \
    runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,                                     \
                                             runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false);        \
    auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);                                     \
    auto [ws_size, executor_handle, release_function] = GEN_EXECUTOR(aclnn_name, __VA_ARGS__);                       \
    if (ws_size == 0) {                                                                                              \
      DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, nullptr, 0, executor_handle, stream_ptr, release_function); \
    } else {                                                                                                         \
      auto work_ptr = std::make_shared<MemBlock>(device_context, ws_size, stream_id);                                \
      DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, work_ptr->ptr_, ws_size, executor_handle, stream_ptr,       \
                             release_function);                                                                      \
    }                                                                                                                \
    static auto sync = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);                \
    if (sync) {                                                                                                      \
      if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {                                        \
        MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;                                              \
      }                                                                                                              \
    } else {                                                                                                         \
      runtime::DeviceAddressUtils::ProcessCrossStreamAddress(aclnn_name, device_context, stream_id, __VA_ARGS__);    \
    }                                                                                                                \
  } while (false)

#define LAUNCH_KERNEL(aclnn_name, ws_ptr, ws_size, executor, stream)                                                  \
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeLaunchTask, \
                                     aclnn_name, false);                                                              \
  MS_LOG(DEBUG) << "launch task start, " << aclnn_name;                                                               \
  RUN_OP_API_SYNC(aclnn_name, ws_ptr, ws_size, executor, stream);                                                     \
  MS_LOG(DEBUG) << "launch task end, " << aclnn_name;

#define LAUNCH_ACLNN_SYNC(aclnn_api, device_context, stream_id, ...)                                          \
  [](const std::string &aclnn_name, const device::DeviceContext *device_context, size_t real_stream_id,       \
     auto &... args) -> auto {                                                                                \
    runtime::Pipeline::Get().WaitForward();                                                                   \
    runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,                              \
                                             runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false); \
    auto stream_ptr = device_context->device_res_manager_->GetStream(real_stream_id);                         \
    auto use_huge_pages = true;                                                                               \
    auto return_values = GEN_EXECUTOR_CUST(aclnn_name, use_huge_pages, args...);                              \
    auto ws_size = std::get<0>(return_values);                                                                \
    auto executor_handle = std::get<1>(return_values);                                                        \
    if (ws_size == 0) {                                                                                       \
      LAUNCH_KERNEL(aclnn_name, nullptr, 0, executor_handle, stream_ptr);                                     \
    } else {                                                                                                  \
      auto work_ptr = std::make_shared<MemBlock>(device_context, ws_size, real_stream_id);                    \
      LAUNCH_KERNEL(aclnn_name, work_ptr->ptr_, ws_size, executor_handle, stream_ptr);                        \
    }                                                                                                         \
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {                                   \
      MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;                                         \
    }                                                                                                         \
    return return_values;                                                                                     \
  }                                                                                                           \
  (#aclnn_api, device_context, stream_id, __VA_ARGS__)

namespace mindspore {
namespace kernel {
namespace pyboost {
struct MemBlock {
  MemBlock(const DeviceContext *device_context, size_t size, uint32_t stream_id) {
    ptr_ = device_context->device_res_manager_->AllocateMemory(size, stream_id);
    if (ptr_ == nullptr) {
      MS_LOG(EXCEPTION) << "Alloc failed, size:" << size << ", stream_id:" << stream_id;
    }
    device_context_ = device_context;
  }
  ~MemBlock() { device_context_->device_res_manager_->FreeMemory(ptr_); }
  void *ptr_;
  const DeviceContext *device_context_;
};
using MemBlockPtr = std::shared_ptr<MemBlock>;
int8_t GetCubeMathType();
std::pair<int64_t, int64_t> UpdateGeneratorState(const tensor::BaseTensorPtr &seed, const tensor::BaseTensorPtr &offset,
                                                 int64_t step = 10);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
