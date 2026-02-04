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
#include <tuple>
#include <list>
#include <unordered_map>
#include <memory>
#include "include/backend/common/device_address_utils.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "kernel/ascend/acl_ir/op_api_exec.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

using ProcessCache = mindspore::device::ascend::ProcessCache;
using CacheTuple = std::tuple<uint64_t, mindspore::device::ascend::aclOpExecutor *, ProcessCache, size_t>;

#define DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, ws_ptr, ws_size, executor, stream, release_func,  \
                               update_func)                                                                  \
  mindspore::runtime::OpExecutor::DispatchLaunchTask(                                                        \
    [dev_ctx = device_context, workspace = ws_ptr, ws_size, executor, stream, release_func, update_func]() { \
      mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,           \
                                                    mindspore::runtime::ProfilerEvent::kPyNativeLaunchTask,  \
                                                    aclnn_name, false);                                      \
      MS_LOG(DEBUG) << "launch task start, " << aclnn_name;                                                  \
      dev_ctx->device_res_manager_->BindDeviceToCurrentThread(false);                                        \
      if (update_func != nullptr) {                                                                          \
        update_func();                                                                                       \
      }                                                                                                      \
      RUN_OP_API_ASYNC(aclnn_name, workspace, ws_size, executor, stream, release_func);                      \
      MS_LOG(DEBUG) << "launch task end, " << aclnn_name;                                                    \
    });

#define DISPATCH_LAUNCH_KERNEL_NO_WS(device_context, aclnn_name, executor, stream, release_func, update_func) \
  mindspore::runtime::OpExecutor::DispatchLaunchTask(                                                         \
    [dev_ctx = device_context, executor, stream, release_func, update_func]() {                               \
      mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,            \
                                                    mindspore::runtime::ProfilerEvent::kPyNativeLaunchTask,   \
                                                    aclnn_name, false);                                       \
      MS_LOG(DEBUG) << "launch task start, " << aclnn_name;                                                   \
      dev_ctx->device_res_manager_->BindDeviceToCurrentThread(false);                                         \
      if (update_func != nullptr) {                                                                           \
        update_func();                                                                                        \
      }                                                                                                       \
      RUN_OP_API_ASYNC(aclnn_name, nullptr, 0, executor, stream, release_func);                               \
      MS_LOG(DEBUG) << "launch task end, " << aclnn_name;                                                     \
    });

#define DISPATCH_LAUNCH_CUSTOM_KERNEL(device_context, aclnn_name, ws_ptr, ws_size, executor, stream, release_func, \
                                      update_func)                                                                 \
  runtime::OpExecutor::DispatchLaunchTask([=]() {                                                                  \
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,                                         \
                                       runtime::ProfilerEvent::kPyNativeLaunchTask, aclnn_name, false);            \
    MS_LOG(DEBUG) << "launch task start, " << aclnn_name;                                                          \
    device_context->device_res_manager_->BindDeviceToCurrentThread(false);                                         \
    if (update_func != nullptr) {                                                                                  \
      update_func();                                                                                               \
    }                                                                                                              \
    RUN_CUSTOM_OP_API_ASYNC(aclnn_name, ws_ptr, ws_size, executor, stream, release_func);                          \
    MS_LOG(DEBUG) << "launch task end, " << aclnn_name;                                                            \
  })

#define GET_EXECUTOR_FOR_PYBOOST(aclnn_api, ...)                                                                  \
  [](const std::string &api_str, const auto &...args) -> auto {                                                   \
    std::unique_lock<std::mutex> lock(mutex_);                                                                    \
    if (MS_UNLIKELY(capacity_ == 0)) {                                                                            \
      auto [ws_size, executor, cache, release_func] = GEN_EXECUTOR(api_str, args...);                             \
      std::function<void()> update_func = nullptr;                                                                \
      return std::make_tuple(ws_size, executor, cache, release_func, update_func);                                \
    }                                                                                                             \
    uint64_t hash_id = mindspore::device::ascend::AclnnHash(api_str, args...);                                    \
    auto iter = hash_map_.find(hash_id);                                                                          \
    if (hash_id != 0 && iter != hash_map_.end()) {                                                                \
      hash_cache_.splice(hash_cache_.begin(), hash_cache_, iter->second);                                         \
      auto cur_run = hash_cache_.front();                                                                         \
      const auto &ws_size = std::get<3>(cur_run);                                                                 \
      const auto &executor = std::get<1>(cur_run);                                                                \
      const auto &cache = std::get<2>(cur_run);                                                                   \
      auto address_list = mindspore::device::ascend::GetTensorAddress(args...);                                   \
      std::function<void()> update_func = [cache, address_list]() -> void {                                       \
        cache(mindspore::device::ascend::ProcessCacheType::kUpdateTensorAddress, address_list);                   \
      };                                                                                                          \
      auto release_func = std::function<void()>(nullptr);                                                         \
      return std::make_tuple(ws_size, executor, cache, release_func, update_func);                                \
    } else {                                                                                                      \
      MS_LOG(INFO) << "Api " << api_str << " miss cache, with hash id:" << hash_id;                               \
      MS_VLOG(mindspore::VLogLevel::VL_ACLNN_OP) << "Api " << api_str << " miss cache, with hash id:" << hash_id; \
      auto [ws_size, executor, cache, fail_cache] = GEN_EXECUTOR_FOR_RESIZE(api_str, args...);                    \
      auto update_func = std::function<void()>(nullptr);                                                          \
      if (MS_LIKELY(hash_id != 0 && !fail_cache)) {                                                               \
        hash_cache_.emplace_front(hash_id, executor, cache, ws_size);                                             \
        hash_map_[hash_id] = hash_cache_.begin();                                                                 \
        if (hash_cache_.size() > capacity_) {                                                                     \
          hash_map_.erase(std::get<0>(hash_cache_.back()));                                                       \
          auto release_func = std::get<2>(hash_cache_.back());                                                    \
          mindspore::runtime::Pipeline::Get().launch_stage()->Wait();                                             \
          release_func(mindspore::device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});               \
          hash_cache_.pop_back();                                                                                 \
        }                                                                                                         \
        auto release_func = std::function<void()>(nullptr);                                                       \
        return std::make_tuple(ws_size, executor, cache, release_func, update_func);                              \
      } else if (MS_UNLIKELY(fail_cache)) {                                                                       \
        std::function<void()> release_func = [cache]() -> void {                                                  \
          cache(mindspore::device::ascend::ProcessCacheType::kReleaseParams, std::vector<std::vector<void *>>{}); \
        };                                                                                                        \
        return std::make_tuple(ws_size, executor, cache, release_func, update_func);                              \
      } else {                                                                                                    \
        MS_LOG(WARNING) << api_str << " cache is available, but hash id is 0, do not use cache.";                 \
        MS_VLOG(mindspore::VLogLevel::VL_ACLNN_OP)                                                                \
          << api_str << " cache is available, but hash id is 0, do not use cache.";                               \
        std::function<void()> release_func = [cache]() -> void {                                                  \
          cache(mindspore::device::ascend::ProcessCacheType::kReleaseParamsAndExecutor,                           \
                std::vector<std::vector<void *>>{});                                                              \
        };                                                                                                        \
        return std::make_tuple(ws_size, executor, cache, release_func, update_func);                              \
      }                                                                                                           \
    }                                                                                                             \
  }(aclnn_api, __VA_ARGS__)

#define LAUNCH_ACLNN(aclnn_api, device_context, stream_id, ...)                                                   \
  do {                                                                                                            \
    static auto simu = mindspore::common::IsCompileSimulation();                                                  \
    if (simu) {                                                                                                   \
      break;                                                                                                      \
    }                                                                                                             \
    static const std::string aclnn_name = #aclnn_api;                                                             \
    static std::unordered_map<uint64_t, std::list<CacheTuple>::iterator> hash_map_;                               \
    static std::list<CacheTuple> hash_cache_;                                                                     \
    static size_t capacity_{1024};                                                                                \
    static std::mutex mutex_;                                                                                     \
    static int64_t capaticy_from_user = mindspore::device::ascend::GetCacheCapaticy();                            \
    static bool not_set_capaticy = true;                                                                          \
    if (capaticy_from_user >= 0 && not_set_capaticy) {                                                            \
      capacity_ = mindspore::LongToSize(capaticy_from_user);                                                      \
      not_set_capaticy = false;                                                                                   \
      MS_LOG(INFO) << "Set aclnn cache queue length of pyboost to " << capacity_;                                 \
      MS_VLOG(mindspore::VLogLevel::VL_ACLNN_OP) << "Set aclnn cache queue length of pyboost to " << capacity_;   \
    }                                                                                                             \
    device_context->device_res_manager_->UseStreamResInCurrentThread(stream_id);                                  \
    mindspore::runtime::ProfilerRecorder aclnn_profiler(mindspore::runtime::ProfilerModule::kPynative,            \
                                                        mindspore::runtime::ProfilerEvent::kPyBoostLaunchAclnn,   \
                                                        aclnn_name, false);                                       \
    auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);                                  \
    auto return_values = GET_EXECUTOR_FOR_PYBOOST(aclnn_name, __VA_ARGS__);                                       \
    auto ws_size = std::get<0>(return_values);                                                                    \
    auto executor_handle = std::get<1>(return_values);                                                            \
    auto release_function = std::get<3>(return_values);                                                           \
    auto update_function = std::get<4>(return_values);                                                            \
    if (ws_size == 0) {                                                                                           \
      DISPATCH_LAUNCH_KERNEL_NO_WS(device_context, aclnn_name, executor_handle, stream_ptr, release_function,     \
                                   update_function);                                                              \
    } else {                                                                                                      \
      auto work_ptr = std::make_shared<mindspore::kernel::pyboost::MemBlock>(device_context, ws_size, stream_id); \
      DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, work_ptr->ptr_, ws_size, executor_handle, stream_ptr,    \
                             release_function, update_function);                                                  \
    }                                                                                                             \
    auto sync = mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking();                                \
    if (sync) {                                                                                                   \
      if (!mindspore::device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {                          \
        MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;                                           \
      }                                                                                                           \
    } else {                                                                                                      \
      mindspore::runtime::DeviceAddressUtils::ProcessCrossStreamAddress(aclnn_name, device_context, stream_id,    \
                                                                        __VA_ARGS__);                             \
    }                                                                                                             \
  } while (false)

#define LAUNCH_KERNEL(device_context, name, ws_ptr, ws_size, executor, stream, update_func)                       \
  runtime::OpExecutor::DispatchLaunchTask(                                                                        \
    [dev_ctx = device_context, aclnn_name = name, workspace = ws_ptr, ws_size, executor, stream, update_func]() { \
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,                                      \
                                         runtime::ProfilerEvent::kPyNativeLaunchTask, aclnn_name, false);         \
      dev_ctx->device_res_manager_->BindDeviceToCurrentThread(false);                                             \
      if (update_func != nullptr) {                                                                               \
        update_func();                                                                                            \
      }                                                                                                           \
      MS_LOG(DEBUG) << "launch task start, " << aclnn_name;                                                       \
      RUN_OP_API_SYNC(aclnn_name, workspace, ws_size, executor, stream);                                          \
      MS_LOG(DEBUG) << "launch task end, " << aclnn_name;                                                         \
    })

#define LAUNCH_KERNEL_NO_WS(device_context, aclnn_name, executor, stream, update_func)                              \
  runtime::OpExecutor::DispatchLaunchTask([dev_ctx = device_context, aclnn_name, executor, stream, update_func]() { \
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,                                          \
                                       runtime::ProfilerEvent::kPyNativeLaunchTask, aclnn_name, false);             \
    dev_ctx->device_res_manager_->BindDeviceToCurrentThread(false);                                                 \
    if (update_func != nullptr) {                                                                                   \
      update_func();                                                                                                \
    }                                                                                                               \
    MS_LOG(DEBUG) << "launch task start, " << aclnn_name;                                                           \
    RUN_OP_API_SYNC(aclnn_name, nullptr, 0, executor, stream);                                                      \
    MS_LOG(DEBUG) << "launch task end, " << aclnn_name;                                                             \
  })

#define LAUNCH_ACLNN_SYNC(aclnn_api, device_context, stream_id, ...)                                          \
  [](const std::string &aclnn_name, const device::DeviceContext *device_context, size_t real_stream_id,       \
     auto &...args) -> auto {                                                                                 \
    static auto simu = common::IsCompileSimulation();                                                         \
    if (simu) {                                                                                               \
      MS_LOG(EXCEPTION) << "For " << aclnn_name << ", the output shape depends on the actual execution,"      \
                        << " and it will affect the accuracy of memory in dryrun mode.";                      \
    }                                                                                                         \
    static std::unordered_map<uint64_t, std::list<CacheTuple>::iterator> hash_map_;                           \
    static std::list<CacheTuple> hash_cache_;                                                                 \
    static size_t capacity_{1024};                                                                            \
    static std::mutex mutex_;                                                                                 \
    static int64_t capaticy_from_user = device::ascend::GetCacheCapaticy();                                   \
    static bool not_set_capaticy = true;                                                                      \
    REGISTER_SYNC_OP(aclnn_name);                                                                             \
    if (capaticy_from_user >= 0 && not_set_capaticy) {                                                        \
      capacity_ = LongToSize(capaticy_from_user);                                                             \
      not_set_capaticy = false;                                                                               \
      MS_LOG(INFO) << "Set aclnn cache queue length of pyboost to " << capacity_;                             \
      MS_VLOG(VL_ACLNN_OP) << "Set aclnn cache queue length of pyboost to " << capacity_;                     \
    }                                                                                                         \
    device_context->device_res_manager_->UseStreamResInCurrentThread(real_stream_id);                         \
    runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,                              \
                                             runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false); \
    auto stream_ptr = device_context->device_res_manager_->GetStream(real_stream_id);                         \
    auto return_values = GET_EXECUTOR_FOR_PYBOOST(aclnn_name, args...);                                       \
    auto ws_size = std::get<0>(return_values);                                                                \
    auto executor_handle = std::get<1>(return_values);                                                        \
    auto update_function = std::get<4>(return_values);                                                        \
    if (ws_size == 0) {                                                                                       \
      LAUNCH_KERNEL_NO_WS(device_context, aclnn_name, executor_handle, stream_ptr, update_function);          \
    } else {                                                                                                  \
      auto work_ptr = std::make_shared<kernel::pyboost::MemBlock>(device_context, ws_size, real_stream_id);   \
      LAUNCH_KERNEL(device_context, aclnn_name, work_ptr->ptr_, ws_size, executor_handle, stream_ptr,         \
                    update_function);                                                                         \
    }                                                                                                         \
    runtime::Pipeline::Get().launch_stage()->Wait();                                                          \
    if (!device::ascend::AscendStreamMng::GetInstance().SyncStream(stream_ptr)) {                             \
      MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;                                         \
    }                                                                                                         \
    const auto &cache_func_ptr = std::get<kIndex2>(return_values);                                            \
    const auto &all_acl_tensor = cache_func_ptr(device::ascend::ProcessCacheType::kGetOutputShape, {});       \
    const auto &release_func = std::get<kIndex3>(return_values);                                              \
    if (release_func) {                                                                                       \
      release_func();                                                                                         \
    }                                                                                                         \
    return all_acl_tensor;                                                                                    \
  }(#aclnn_api, device_context, stream_id, __VA_ARGS__)

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

int8_t GetCubeMathType(bool use_hf32 = false);
bool IsAllowMatmulHF32();
bool IsAllowConvHF32();
std::pair<int64_t, int64_t> GetGeneratorState(const tensor::TensorPtr &seed, const tensor::TensorPtr &offset);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
