/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <utility>
#include "ops/base_operator.h"
#include "include/op_def.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/utils/utils.h"
#include "tools/profiler/profiler.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "kernel/ascend/acl_ir/op_api_exec.h"
#include "kernel/ascend/acl_ir/op_api_util.h"
#include "utils/ms_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_utils.h"
#include "kernel/ascend/visible.h"

namespace mindspore {
namespace kernel {
using aclTensor = device::ascend::aclTensor;
using aclOpExecutor = device::ascend::aclOpExecutor;
using CallBackFunc = std::function<void()>;
using OpApiUtil = device::ascend::OpApiUtil;
using AclUtil = device::ascend::AclUtil;
using ProcessCache = device::ascend::ProcessCache;
using CacheTuple = std::tuple<uint64_t, aclOpExecutor *, ProcessCache, size_t>;

#define DEFINE_GET_WORKSPACE_FOR_OPS(OP_TYPE, FUNC_NAME)                                                             \
  std::string op_type_##FUNC_NAME##_ = #OP_TYPE;                                                                     \
  uint64_t hash_id_##FUNC_NAME##_{0};                                                                                \
  aclOpExecutor *executor_##FUNC_NAME##_{nullptr};                                                                   \
  CallBackFunc release_func_##FUNC_NAME##_{nullptr};                                                                 \
  template <typename... Args>                                                                                        \
  void GetWorkspaceForResize##FUNC_NAME(const Args &...args) {                                                       \
    size_t cur_workspace = 0;                                                                                        \
    if (capacity_ == 0) {                                                                                            \
      if (is_dynamic_) {                                                                                             \
        hash_id_##FUNC_NAME##_ = 0;                                                                                  \
      } else {                                                                                                       \
        std::tie(cur_workspace, std::ignore, std::ignore) = GEN_EXECUTOR_CUST(op_type_##FUNC_NAME##_, args...);      \
        if (cur_workspace != 0) {                                                                                    \
          ops_workspace_size_map_[#FUNC_NAME] = {ops_workspace_size_idx_, cur_workspace};                            \
          ++ops_workspace_size_idx_;                                                                                 \
          (void)workspace_size_list_.emplace_back(cur_workspace);                                                    \
        }                                                                                                            \
      }                                                                                                              \
      return;                                                                                                        \
    }                                                                                                                \
    hash_id_##FUNC_NAME##_ = device::ascend::AclnnHash(op_type_##FUNC_NAME##_, args...);                             \
    auto iter = hash_map_.find(hash_id_##FUNC_NAME##_);                                                              \
    if (iter != hash_map_.end()) {                                                                                   \
      MS_VLOG(VL_ACLNN_OP) << "Op " << op_type_##FUNC_NAME##_                                                        \
                           << " hit cache with hash id: " << hash_id_##FUNC_NAME##_;                                 \
      hash_cache_.splice(hash_cache_.begin(), hash_cache_, iter->second);                                            \
      cur_workspace = std::get<3>(hash_cache_.front());                                                              \
    } else {                                                                                                         \
      MS_VLOG(VL_ACLNN_OP) << "op " << op_type_##FUNC_NAME##_                                                        \
                           << " miss cache with hash id: " << hash_id_##FUNC_NAME##_;                                \
      auto [workspace, executor, cache, fail_cache] = GEN_EXECUTOR_FOR_RESIZE(op_type_##FUNC_NAME##_, args...);      \
      cur_workspace = workspace;                                                                                     \
      if (!fail_cache) {                                                                                             \
        hash_cache_.emplace_front(hash_id_##FUNC_NAME##_, executor, cache, workspace);                               \
        hash_map_[hash_id_##FUNC_NAME##_] = hash_cache_.begin();                                                     \
        if (hash_cache_.size() > capacity_) {                                                                        \
          hash_map_.erase(std::get<0>(hash_cache_.back()));                                                          \
          auto release_func = std::get<2>(hash_cache_.back());                                                       \
          release_func(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});                             \
          hash_cache_.pop_back();                                                                                    \
        }                                                                                                            \
      } else {                                                                                                       \
        hash_id_##FUNC_NAME##_ = 0;                                                                                  \
        cache(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});                                      \
      }                                                                                                              \
    }                                                                                                                \
                                                                                                                     \
    if (cur_workspace != 0) {                                                                                        \
      ops_workspace_size_map_[#FUNC_NAME] = {ops_workspace_size_idx_, cur_workspace};                                \
      ++ops_workspace_size_idx_;                                                                                     \
      (void)workspace_size_list_.emplace_back(cur_workspace);                                                        \
    }                                                                                                                \
  }                                                                                                                  \
                                                                                                                     \
  template <typename... Args>                                                                                        \
  std::pair<aclOpExecutor *, std::function<void()>> GetExecutor##FUNC_NAME(const Args &...args) {                    \
    auto iter = hash_map_.find(hash_id_##FUNC_NAME##_);                                                              \
    if (hash_id_##FUNC_NAME##_ == 0 || iter == hash_map_.end()) {                                                    \
      aclOpExecutor *executor;                                                                                       \
      std::function<void()> release_func;                                                                            \
      std::tie(std::ignore, executor, release_func, hash_id_##FUNC_NAME##_, std::ignore) =                           \
        GEN_EXECUTOR_BOOST(op_type_##FUNC_NAME##_, hash_id_##FUNC_NAME##_, args...);                                 \
      return std::make_pair(executor, release_func);                                                                 \
    }                                                                                                                \
    const auto &cur_run = *(iter->second);                                                                           \
    UPDATE_TENSOR_FOR_LAUNCH(std::get<2>(cur_run), args...);                                                         \
    const auto &executor = std::get<1>(cur_run);                                                                     \
    return std::make_pair(executor, nullptr);                                                                        \
  }                                                                                                                  \
                                                                                                                     \
  template <typename... Args>                                                                                        \
  void RunOp##FUNC_NAME(void *stream_ptr, const std::vector<KernelTensor *> &workspace, const Args &...args) {       \
    if (capacity_ == 0) {                                                                                            \
      size_t ws_size = 0;                                                                                            \
      std::tie(ws_size, executor_##FUNC_NAME##_, release_func_##FUNC_NAME##_, hash_id_##FUNC_NAME##_, std::ignore) = \
        GEN_EXECUTOR_BOOST(op_type_##FUNC_NAME##_, hash_id_##FUNC_NAME##_, args...);                                 \
      if (ws_size == 0) {                                                                                            \
        RUN_OP_API_ASYNC(op_type_##FUNC_NAME##_, nullptr, 0, executor_##FUNC_NAME##_, stream_ptr,                    \
                         release_func_##FUNC_NAME##_);                                                               \
      } else {                                                                                                       \
        if (is_dynamic_) {                                                                                           \
          static device::DeviceContext *device_context =                                                             \
            device::DeviceContextManager::GetInstance().GetDeviceContext("Ascend").get();                            \
          auto ws_ptr = std::make_shared<kernel::MemBlock>(device_context, ws_size, stream_ptr);                     \
          RUN_OP_API_ASYNC(op_type_##FUNC_NAME##_, ws_ptr->ptr_, ws_size, executor_##FUNC_NAME##_, stream_ptr,       \
                           release_func_##FUNC_NAME##_);                                                             \
        } else {                                                                                                     \
          const auto &iter = ops_workspace_size_map_.find(#FUNC_NAME);                                               \
          if (iter == ops_workspace_size_map_.end()) {                                                               \
            MS_LOG(EXCEPTION) << "Failed to get workspace size for " << #FUNC_NAME;                                  \
          }                                                                                                          \
          auto workspace_size_idx = iter->second.first;                                                              \
          auto workspace_size = iter->second.second;                                                                 \
          if (workspace.empty() || workspace.size() <= workspace_size_idx) {                                         \
            MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";                                             \
          }                                                                                                          \
          auto workspace_tensor = workspace[workspace_size_idx];                                                     \
          RUN_OP_API_ASYNC(op_type_##FUNC_NAME##_, workspace_tensor->device_ptr(), workspace_size,                   \
                           executor_##FUNC_NAME##_, stream_ptr, release_func_##FUNC_NAME##_);                        \
        }                                                                                                            \
      }                                                                                                              \
      return;                                                                                                        \
    }                                                                                                                \
    auto [executor, release_func] = GetExecutor##FUNC_NAME(args...);                                                 \
    const auto &iter = ops_workspace_size_map_.find(#FUNC_NAME);                                                     \
    if (iter == ops_workspace_size_map_.end()) {                                                                     \
      RUN_OP_API_ASYNC(op_type_##FUNC_NAME##_, nullptr, 0, executor, stream_ptr, release_func);                      \
    } else {                                                                                                         \
      auto workspace_size_idx = iter->second.first;                                                                  \
      auto workspace_size = iter->second.second;                                                                     \
      if (workspace.empty() || workspace.size() <= workspace_size_idx) {                                             \
        MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";                                                 \
      }                                                                                                              \
      auto workspace_tensor = workspace[workspace_size_idx];                                                         \
      if (workspace_tensor->size() != workspace_size) {                                                              \
        MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"         \
                          << workspace_size << ", but get " << workspace_tensor->size();                             \
      }                                                                                                              \
      RUN_OP_API_ASYNC(op_type_##FUNC_NAME##_, workspace_tensor->device_ptr(), workspace_size, executor, stream_ptr, \
                       release_func);                                                                                \
    }                                                                                                                \
  }

#define DEFINE_GET_WORKSPACE_FOR_RESIZE()                                                                       \
  template <typename... Args>                                                                                   \
  void GetWorkspaceForResize(const Args &...args) {                                                             \
    size_t cur_workspace = 0;                                                                                   \
    if (capacity_ == 0) {                                                                                       \
      if (is_dynamic_) {                                                                                        \
        hash_id_ = 0;                                                                                           \
      } else {                                                                                                  \
        std::tie(cur_workspace, std::ignore, std::ignore) = GEN_EXECUTOR_CUST(op_type_, args...);               \
        if (cur_workspace != 0) {                                                                               \
          std::vector<size_t> workspace_size_list = {cur_workspace};                                            \
          SetWorkspaceSizeList(workspace_size_list);                                                            \
        }                                                                                                       \
      }                                                                                                         \
      return;                                                                                                   \
    }                                                                                                           \
    hash_id_ = device::ascend::AclnnHash(op_type_, args...);                                                    \
    auto iter = hash_map_.find(hash_id_);                                                                       \
    if (iter != hash_map_.end()) {                                                                              \
      MS_VLOG(VL_ACLNN_OP) << "op " << op_type_ << " hit cache with hash id: " << hash_id_;                     \
      hash_cache_.splice(hash_cache_.begin(), hash_cache_, iter->second);                                       \
      cur_workspace = std::get<3>(hash_cache_.front());                                                         \
    } else {                                                                                                    \
      MS_VLOG(VL_ACLNN_OP) << "op " << op_type_ << " miss cache with hash id: " << hash_id_;                    \
      auto [workspace, executor, cache, fail_cache] = GEN_EXECUTOR_FOR_RESIZE(op_type_, args...);               \
      cur_workspace = workspace;                                                                                \
      if (!fail_cache) {                                                                                        \
        hash_cache_.emplace_front(hash_id_, executor, cache, workspace);                                        \
        hash_map_[hash_id_] = hash_cache_.begin();                                                              \
        if (hash_cache_.size() > capacity_) {                                                                   \
          hash_map_.erase(std::get<0>(hash_cache_.back()));                                                     \
          auto release_func = std::get<2>(hash_cache_.back());                                                  \
          release_func(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});                        \
          hash_cache_.pop_back();                                                                               \
        }                                                                                                       \
      } else {                                                                                                  \
        hash_id_ = 0;                                                                                           \
        cache(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});                                 \
      }                                                                                                         \
    }                                                                                                           \
                                                                                                                \
    if (cur_workspace != 0) {                                                                                   \
      std::vector<size_t> workspace_size_list = {cur_workspace};                                                \
      SetWorkspaceSizeList(workspace_size_list);                                                                \
    }                                                                                                           \
  }                                                                                                             \
                                                                                                                \
  template <typename... Args>                                                                                   \
  std::pair<aclOpExecutor *, std::function<void()>> GetExecutor(const Args &...args) {                          \
    auto iter = hash_map_.find(hash_id_);                                                                       \
    if (hash_id_ == 0 || iter == hash_map_.end()) {                                                             \
      aclOpExecutor *executor;                                                                                  \
      std::function<void()> release_func;                                                                       \
      std::tie(std::ignore, executor, release_func, hash_id_, std::ignore) =                                    \
        GEN_EXECUTOR_BOOST(op_type_, hash_id_, args...);                                                        \
      return std::make_pair(executor, release_func);                                                            \
    }                                                                                                           \
    const auto &cur_run = *(iter->second);                                                                      \
    UPDATE_TENSOR_FOR_LAUNCH(std::get<2>(cur_run), args...);                                                    \
    const auto &executor = std::get<1>(cur_run);                                                                \
    return std::make_pair(executor, nullptr);                                                                   \
  }                                                                                                             \
                                                                                                                \
  template <typename... Args>                                                                                   \
  void RunOp(void *stream_ptr, const std::vector<KernelTensor *> &workspace, const Args &...args) {             \
    if (capacity_ == 0) {                                                                                       \
      size_t ws_size = 0;                                                                                       \
      std::tie(ws_size, executor_, release_func_, hash_id_, std::ignore) =                                      \
        GEN_EXECUTOR_BOOST(op_type_, hash_id_, args...);                                                        \
      if (ws_size == 0) {                                                                                       \
        RUN_OP_API_ASYNC(op_type_, nullptr, 0, executor_, stream_ptr, release_func_);                           \
      } else {                                                                                                  \
        if (is_dynamic_) {                                                                                      \
          static device::DeviceContext *device_context =                                                        \
            device::DeviceContextManager::GetInstance().GetDeviceContext("Ascend").get();                       \
          auto ws_ptr = std::make_shared<kernel::MemBlock>(device_context, ws_size, stream_ptr);                \
          RUN_OP_API_ASYNC(op_type_, ws_ptr->ptr_, ws_size, executor_, stream_ptr, release_func_);              \
        } else {                                                                                                \
          auto ws_tensor = workspace[0];                                                                        \
          RUN_OP_API_ASYNC(op_type_, ws_tensor->device_ptr(), ws_size, executor_, stream_ptr, release_func_);   \
        }                                                                                                       \
      }                                                                                                         \
      return;                                                                                                   \
    }                                                                                                           \
    auto [executor, release_func] = GetExecutor(args...);                                                       \
    if (workspace_size_list_.empty()) {                                                                         \
      RUN_OP_API_ASYNC(op_type_, nullptr, 0, executor, stream_ptr, release_func);                               \
    } else {                                                                                                    \
      if (workspace.empty()) {                                                                                  \
        MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";                                            \
      }                                                                                                         \
      auto workspace_tensor = workspace[0];                                                                     \
      if (workspace_tensor->size() != workspace_size_list_[0]) {                                                \
        MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"    \
                          << workspace_size_list_[0] << ", but get " << workspace_tensor->size();               \
      }                                                                                                         \
      RUN_OP_API_ASYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor, stream_ptr, \
                       release_func);                                                                           \
    }                                                                                                           \
  }                                                                                                             \
                                                                                                                \
  template <typename... Args>                                                                                   \
  std::tuple<aclOpExecutor *, ProcessCache, std::function<void()>> GetSyncExecutor(const Args &...args) {       \
    auto iter = hash_map_.find(hash_id_);                                                                       \
    if (hash_id_ == 0 || iter == hash_map_.end()) {                                                             \
      aclOpExecutor *executor;                                                                                  \
      ProcessCache cache_func_ptr;                                                                              \
      std::function<void()> release_func;                                                                       \
      std::tie(std::ignore, executor, cache_func_ptr, release_func) = GEN_EXECUTOR(op_type_, args...);          \
      return std::make_tuple(executor, cache_func_ptr, release_func);                                           \
    }                                                                                                           \
    const auto &cur_run = *(iter->second);                                                                      \
    const auto &cache_func_ptr = std::get<2>(cur_run);                                                          \
    UPDATE_TENSOR_FOR_LAUNCH(cache_func_ptr, args...);                                                          \
    const auto &executor = std::get<1>(cur_run);                                                                \
    return std::make_tuple(executor, cache_func_ptr, nullptr);                                                  \
  }                                                                                                             \
                                                                                                                \
  template <typename... Args>                                                                                   \
  std::vector<ShapeVector> RunOpSync(void *stream_ptr, const std::vector<KernelTensor *> &workspace,            \
                                     const Args &...args) {                                                     \
    REGISTER_SYNC_OP(op_type_);                                                                                 \
    ProcessCache cache_func_ptr;                                                                                \
    std::function<void()> release_func;                                                                         \
    if (capacity_ == 0) {                                                                                       \
      size_t ws_size = 0;                                                                                       \
      std::tie(ws_size, executor_, cache_func_ptr, release_func) = GEN_EXECUTOR(op_type_, args...);             \
      if (ws_size == 0) {                                                                                       \
        RUN_OP_API_SYNC(op_type_, nullptr, 0, executor_, stream_ptr);                                           \
      } else {                                                                                                  \
        if (is_dynamic_) {                                                                                      \
          static device::DeviceContext *device_context =                                                        \
            device::DeviceContextManager::GetInstance().GetDeviceContext("Ascend").get();                       \
          auto ws_ptr = std::make_shared<kernel::MemBlock>(device_context, ws_size, stream_ptr);                \
          RUN_OP_API_SYNC(op_type_, ws_ptr->ptr_, ws_size, executor_, stream_ptr);                              \
        } else {                                                                                                \
          auto ws_tensor = workspace[0];                                                                        \
          RUN_OP_API_SYNC(op_type_, ws_tensor->device_ptr(), ws_size, executor_, stream_ptr);                   \
        }                                                                                                       \
      }                                                                                                         \
    } else {                                                                                                    \
      std::tie(executor_, cache_func_ptr, release_func) = GetSyncExecutor(args...);                             \
      if (workspace_size_list_.empty()) {                                                                       \
        RUN_OP_API_SYNC(op_type_, nullptr, 0, executor_, stream_ptr);                                           \
      } else {                                                                                                  \
        if (workspace.empty()) {                                                                                \
          MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";                                          \
        }                                                                                                       \
        auto ws_tensor = workspace[0];                                                                          \
        if (ws_tensor->size() != workspace_size_list_[0]) {                                                     \
          MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"  \
                            << workspace_size_list_[0] << ", but get " << ws_tensor->size();                    \
        }                                                                                                       \
        RUN_OP_API_SYNC(op_type_, ws_tensor->device_ptr(), workspace_size_list_[0], executor_, stream_ptr);     \
      }                                                                                                         \
    }                                                                                                           \
    auto ret = CALL_ASCEND_API(aclrtSynchronizeStream, stream_ptr);                                             \
    if (ret != 0) {                                                                                             \
      MS_LOG(EXCEPTION) << "Sync stream " << op_type_ << " error: " << CALL_ASCEND_API(aclGetRecentErrMsg);     \
    }                                                                                                           \
    const auto &all_acl_tensor = cache_func_ptr(device::ascend::ProcessCacheType::kGetOutputShape, {});         \
    if (release_func) {                                                                                         \
      release_func();                                                                                           \
    }                                                                                                           \
    return all_acl_tensor;                                                                                      \
  }

class EmptyKernelTensor {
 public:
  EmptyKernelTensor() { tensor_ = new KernelTensor(); }
  EmptyKernelTensor(TypeId type_id, TypeId dtype_id) {
    if (type_id == kObjectTypeTensorType) {
      tensor_ = new KernelTensor();
      auto tensor_shape = std::make_shared<abstract::TensorShape>();
      tensor_shape->SetShapeVector({0});
      tensor_->SetType(std::make_shared<TensorType>(TypeIdToType(dtype_id)));
      tensor_->SetShape(tensor_shape);
    }
  }
  ~EmptyKernelTensor() { delete tensor_; }
  KernelTensor *get() const { return tensor_; }

 private:
  KernelTensor *tensor_;
};

struct MemBlock {
  MemBlock(device::DeviceContext *device_context, size_t size, void *stream) {
    auto stream_id = device::ascend::AscendStreamMng::GetInstance().GetStreamId(stream);
    ptr_ = device_context->device_res_manager_->AllocateMemory(size, stream_id);
    if (ptr_ == nullptr) {
      MS_LOG(EXCEPTION) << "Alloc failed, size:" << size << ", stream_id:" << stream_id;
    }
    device_context_ = device_context;
  }
  ~MemBlock() { device_context_->device_res_manager_->FreeMemory(ptr_); }
  void *ptr_;
  const device::DeviceContext *device_context_;
};

class OPS_ASCEND_API AclnnKernelMod : public KernelMod {
 public:
  explicit AclnnKernelMod(std::string &&op_type) : op_type_(std::move(op_type)) {
    auto capaticy_from_user = device::ascend::GetCacheCapaticy();
    if (capaticy_from_user >= 0) {
      capacity_ = LongToSize(capaticy_from_user);
      MS_VLOG(VL_ACLNN_OP) << "Set aclnn cache queue length of kbyk to " << capacity_;
      MS_LOG(INFO) << "Set aclnn cache queue length of kbyk to " << capacity_;
    }
    device_context_ = device::DeviceContextManager::GetInstance().GetDeviceContext("Ascend").get();
  }
  ~AclnnKernelMod();

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  virtual void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  }

  virtual bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                      const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  void ResetDeivceAddress(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {}

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override;
  bool IsNeedUpdateOutputShapeAndSize() override { return false; }
  std::vector<KernelAttr> GetOpSupport() override { MS_LOG(EXCEPTION) << "This interface is not support in aclnn."; }
  void set_fullname(const std::string &fullname) override { fullname_ = fullname; }

  template <typename... Args>
  void UpdateWorkspace(const std::tuple<Args...> &args) {
    auto real_workspace_size = static_cast<size_t>(std::get<0>(args));
    if (real_workspace_size != 0) {
      std::vector<size_t> workspace_size_list = {real_workspace_size};
      SetWorkspaceSizeList(workspace_size_list);
    }

    constexpr size_t kBoostGeneratorSize = 5;
    if constexpr (std::tuple_size_v<std::tuple<Args...>> == kBoostGeneratorSize) {
      hash_id_ = std::get<kHashIdIndex>(args);
    }
  }

  void SetDynamic(bool is_dynamic) { is_dynamic_ = is_dynamic; }

  void ClearOpsWorkSpaceList() {
    ops_workspace_size_idx_ = 0;
    ops_workspace_size_map_.clear();
    workspace_size_list_.clear();
  }

 protected:
  template <typename T>
  T GetRequiredAttr(const std::string &attr_name) {
    auto attr_value = primitive_->GetAttr(attr_name);
    return GetValue<T>(attr_value);
  }

  aclOpExecutor *executor_{nullptr};
  CallBackFunc release_func_{nullptr};
  std::string op_type_;
  uint64_t hash_id_{0};
  std::unordered_map<std::string, std::pair<size_t, size_t>> ops_workspace_size_map_;
  size_t ops_workspace_size_idx_{0};
  static bool is_dynamic_;
  std::unordered_map<uint64_t, std::list<CacheTuple>::iterator> hash_map_;
  std::list<CacheTuple> hash_cache_;
  size_t capacity_{64};
  std::string fullname_;
  const device::DeviceContext *device_context_;
  uint32_t stream_id_{kInValidStreamIndex};
  static constexpr size_t kWsSizeIndex = 0;
  static constexpr size_t kHashIdIndex = 3;
};

using AclnnKernelModPtr = std::shared_ptr<AclnnKernelMod>;
using AclnnKernelModPtrList = std::vector<AclnnKernelModPtr>;

#define REGISTER_ACLNN_CLASS(TYPE)                                                                                   \
  template <size_t N>                                                                                                \
  class Aclnn##TYPE##KernelMod : public AclnnKernelMod {                                                             \
   public:                                                                                                           \
    explicit Aclnn##TYPE##KernelMod(std::string &&op_type) : AclnnKernelMod(std::move(op_type)) {}                   \
    ~Aclnn##TYPE##KernelMod() = default;                                                                             \
    void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,                                                 \
                          const std::vector<KernelTensor *> &outputs) override {                                     \
      const auto &res_tuple = GetKernelTuple<N>(inputs, outputs);                                                    \
      std::apply([this](const auto &...args) { GetWorkspaceForResize(args...); }, res_tuple);                        \
    }                                                                                                                \
    bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,             \
                const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {                             \
      CallRun(stream_ptr, workspace, inputs, outputs);                                                               \
      return true;                                                                                                   \
    }                                                                                                                \
                                                                                                                     \
   private:                                                                                                          \
    template <typename... Ts>                                                                                        \
    void CallRun(void *stream_ptr, const std::vector<KernelTensor *> &workspace, const std::vector<Ts> &...vecs) {   \
      const auto &res_tuple = GetKernelTuple<N>(vecs...);                                                            \
      std::apply(                                                                                                    \
        [this, stream_ptr, &workspace](const auto &...args) { return this->RunOp(stream_ptr, workspace, args...); }, \
        res_tuple);                                                                                                  \
    }                                                                                                                \
                                                                                                                     \
    DEFINE_GET_WORKSPACE_FOR_RESIZE()                                                                                \
  }

#define MS_ACLNN_KERNEL_FACTORY_REG(NAME, DERIVE_CLASS) MS_KERNEL_FACTORY_REG(AclnnKernelMod, NAME, DERIVE_CLASS)
#define MS_ACLNN_COMMON_KERNEL_FACTORY_REG(NAME, TYPE, N)                     \
  REGISTER_ACLNN_CLASS(NAME);                                                 \
  static const KernelRegistrar<AclnnKernelMod> g_##NAME##_AclnnKernelMod_reg( \
    #NAME, []() { return std::make_shared<Aclnn##NAME##KernelMod<N>>(#TYPE); })
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_KERNEL_MOD_H_
