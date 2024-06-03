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
#include "ops/op_def.h"
#include "kernel/kernel.h"
#include "plugin/factory/ms_factory.h"
#include "include/common/utils/utils.h"
#include "include/common/profiler.h"
#include "runtime/pynative/op_runtime_info.h"
#include "transform/acl_ir/acl_convert.h"
#include "transform/acl_ir/op_api_exec.h"
#include "transform/acl_ir/op_api_util.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"

namespace mindspore {
namespace kernel {
using aclTensor = transform::aclTensor;
using aclOpExecutor = transform::aclOpExecutor;
using CallBackFunc = std::function<void()>;
using OpApiUtil = transform::OpApiUtil;
using AclUtil = transform::AclUtil;
using ProcessCache = transform::ProcessCache;
using CacheTuple = std::tuple<uint64_t, aclOpExecutor *, ProcessCache, size_t>;

#define DEFINE_GET_WORKSPACE_FOR_OPS(OP_TYPE, FUNC_NAME)                                                              \
  std::string op_type_##FUNC_NAME##_ = #OP_TYPE;                                                                      \
  template <typename... Args>                                                                                         \
  void GetWorkspaceForResize##FUNC_NAME(const Args &... args) {                                                       \
    hash_id_ = transform::AclnnHash(op_type_##FUNC_NAME##_, args...);                                                 \
    size_t cur_workspace = 0;                                                                                         \
    if (hash_map_.count(hash_id_)) {                                                                                  \
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel,                                            \
                                         runtime::ProfilerEvent::kAclnnHitCacheStage1, op_type_##FUNC_NAME##_);       \
      hash_cache_.splice(hash_cache_.begin(), hash_cache_, hash_map_[hash_id_]);                                      \
      cur_workspace = std::get<3>(hash_cache_.front());                                                               \
    } else {                                                                                                          \
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel,                                            \
                                         runtime::ProfilerEvent::kAclnnMissCacheStage1, op_type_##FUNC_NAME##_);      \
      auto [workspace, executor, cache, fail_cache] = GEN_EXECUTOR_FOR_RESIZE(op_type_##FUNC_NAME##_, args...);       \
      cur_workspace = workspace;                                                                                      \
      if (!fail_cache) {                                                                                              \
        hash_cache_.emplace_front(hash_id_, executor, cache, workspace);                                              \
        hash_map_[hash_id_] = hash_cache_.begin();                                                                    \
      } else {                                                                                                        \
        hash_id_ = 0;                                                                                                 \
        cache(true, {});                                                                                              \
      }                                                                                                               \
    }                                                                                                                 \
    if (hash_cache_.size() > capacity_) {                                                                             \
      hash_map_.erase(std::get<0>(hash_cache_.back()));                                                               \
      auto release_func = std::get<2>(hash_cache_.back());                                                            \
      release_func(true, {});                                                                                         \
      hash_cache_.pop_back();                                                                                         \
    }                                                                                                                 \
                                                                                                                      \
    if (cur_workspace != 0) {                                                                                         \
      std::vector<size_t> workspace_size_list = {cur_workspace};                                                      \
      SetWorkspaceSizeList(workspace_size_list);                                                                      \
    }                                                                                                                 \
  }                                                                                                                   \
                                                                                                                      \
  template <typename... Args>                                                                                         \
  std::pair<aclOpExecutor *, std::function<void()>> GetExecutor##FUNC_NAME(const Args &... args) {                    \
    if (hash_id_ == 0 || !hash_map_.count(hash_id_)) {                                                                \
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel,                                            \
                                         runtime::ProfilerEvent::kAclnnMissCacheStage2, op_type_##FUNC_NAME##_);      \
      aclOpExecutor *executor;                                                                                        \
      std::function<void()> release_func;                                                                             \
      std::tie(std::ignore, executor, release_func) = GEN_EXECUTOR(op_type_##FUNC_NAME##_, args...);                  \
      return std::make_pair(executor, release_func);                                                                  \
    }                                                                                                                 \
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kAclnnUpdateAddress, \
                                       op_type_##FUNC_NAME##_);                                                       \
    auto cur_run = *hash_map_[hash_id_];                                                                              \
    UPDATE_TENSOR_FOR_LAUNCH(std::get<2>(cur_run), args...);                                                          \
    auto executor = std::get<1>(cur_run);                                                                             \
    return std::make_pair(executor, nullptr);                                                                         \
  }                                                                                                                   \
                                                                                                                      \
  template <typename... Args>                                                                                         \
  void RunOp##FUNC_NAME(void *stream_ptr, const std::vector<KernelTensor *> &workspace, const Args &... args) {       \
    auto [executor, release_func] = GetExecutor(args...);                                                             \
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kAclnnRunOp,         \
                                       op_type_##FUNC_NAME##_);                                                       \
    if (workspace_size_list_.empty()) {                                                                               \
      RUN_OP_API_ASYNC(op_type_##FUNC_NAME##_, nullptr, 0, executor, stream_ptr, release_func);                       \
    } else {                                                                                                          \
      if (workspace.empty()) {                                                                                        \
        MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";                                                  \
      }                                                                                                               \
      auto workspace_tensor = workspace[0];                                                                           \
      if (workspace_tensor->size() != workspace_size_list_[0]) {                                                      \
        MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"          \
                          << workspace_size_list_[0] << ", but get " << workspace_tensor->size();                     \
      }                                                                                                               \
      RUN_OP_API_ASYNC(op_type_##FUNC_NAME##_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor,     \
                       stream_ptr, release_func);                                                                     \
    }                                                                                                                 \
  }

#define DEFINE_GET_WORKSPACE_FOR_RESIZE()                                                                             \
  template <typename... Args>                                                                                         \
  void GetWorkspaceForResize(const Args &... args) {                                                                  \
    hash_id_ = transform::AclnnHash(op_type_, args...);                                                               \
    size_t cur_workspace = 0;                                                                                         \
    if (hash_map_.count(hash_id_)) {                                                                                  \
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel,                                            \
                                         runtime::ProfilerEvent::kAclnnHitCacheStage1, op_type_);                     \
      hash_cache_.splice(hash_cache_.begin(), hash_cache_, hash_map_[hash_id_]);                                      \
      cur_workspace = std::get<3>(hash_cache_.front());                                                               \
    } else {                                                                                                          \
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel,                                            \
                                         runtime::ProfilerEvent::kAclnnMissCacheStage1, op_type_);                    \
      auto [workspace, executor, cache, fail_cache] = GEN_EXECUTOR_FOR_RESIZE(op_type_, args...);                     \
      cur_workspace = workspace;                                                                                      \
      if (!fail_cache) {                                                                                              \
        hash_cache_.emplace_front(hash_id_, executor, cache, workspace);                                              \
        hash_map_[hash_id_] = hash_cache_.begin();                                                                    \
      } else {                                                                                                        \
        hash_id_ = 0;                                                                                                 \
        cache(true, {});                                                                                              \
      }                                                                                                               \
    }                                                                                                                 \
    if (hash_cache_.size() > capacity_) {                                                                             \
      hash_map_.erase(std::get<0>(hash_cache_.back()));                                                               \
      auto release_func = std::get<2>(hash_cache_.back());                                                            \
      release_func(true, {});                                                                                         \
      hash_cache_.pop_back();                                                                                         \
    }                                                                                                                 \
                                                                                                                      \
    if (cur_workspace != 0) {                                                                                         \
      std::vector<size_t> workspace_size_list = {cur_workspace};                                                      \
      SetWorkspaceSizeList(workspace_size_list);                                                                      \
    }                                                                                                                 \
  }                                                                                                                   \
                                                                                                                      \
  template <typename... Args>                                                                                         \
  std::pair<aclOpExecutor *, std::function<void()>> GetExecutor(const Args &... args) {                               \
    if (hash_id_ == 0 || !hash_map_.count(hash_id_)) {                                                                \
      runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel,                                            \
                                         runtime::ProfilerEvent::kAclnnMissCacheStage2, op_type_);                    \
      aclOpExecutor *executor;                                                                                        \
      std::function<void()> release_func;                                                                             \
      std::tie(std::ignore, executor, release_func) = GEN_EXECUTOR(op_type_, args...);                                \
      return std::make_pair(executor, release_func);                                                                  \
    }                                                                                                                 \
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kAclnnUpdateAddress, \
                                       op_type_);                                                                     \
    auto cur_run = *hash_map_[hash_id_];                                                                              \
    UPDATE_TENSOR_FOR_LAUNCH(std::get<2>(cur_run), args...);                                                          \
    auto executor = std::get<1>(cur_run);                                                                             \
    return std::make_pair(executor, nullptr);                                                                         \
  }                                                                                                                   \
                                                                                                                      \
  template <typename... Args>                                                                                         \
  void RunOp(void *stream_ptr, const std::vector<KernelTensor *> &workspace, const Args &... args) {                  \
    auto [executor, release_func] = GetExecutor(args...);                                                             \
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kAclnnRunOp,         \
                                       op_type_);                                                                     \
    if (workspace_size_list_.empty()) {                                                                               \
      RUN_OP_API_ASYNC(op_type_, nullptr, 0, executor, stream_ptr, release_func);                                     \
    } else {                                                                                                          \
      if (workspace.empty()) {                                                                                        \
        MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";                                                  \
      }                                                                                                               \
      auto workspace_tensor = workspace[0];                                                                           \
      if (workspace_tensor->size() != workspace_size_list_[0]) {                                                      \
        MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"          \
                          << workspace_size_list_[0] << ", but get " << workspace_tensor->size();                     \
      }                                                                                                               \
      RUN_OP_API_ASYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor, stream_ptr,       \
                       release_func);                                                                                 \
    }                                                                                                                 \
  }                                                                                                                   \
                                                                                                                      \
  template <typename... Args>                                                                                         \
  void RunOpSync(void *stream_ptr, const std::vector<KernelTensor *> &workspace, const Args &... args) {              \
    aclOpExecutor *executor = executor_;                                                                              \
    if (executor == nullptr) {                                                                                        \
      std::tie(executor, std::ignore) = GetExecutor(args...);                                                         \
    }                                                                                                                 \
    if (workspace_size_list_.empty()) {                                                                               \
      RUN_OP_API_SYNC(op_type_, nullptr, 0, executor, stream_ptr);                                                    \
    } else {                                                                                                          \
      if (workspace.empty()) {                                                                                        \
        MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";                                                  \
      }                                                                                                               \
      auto workspace_tensor = workspace[0];                                                                           \
      if (workspace_tensor->size() != workspace_size_list_[0]) {                                                      \
        MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"          \
                          << workspace_size_list_[0] << ", but get " << workspace_tensor->size();                     \
      }                                                                                                               \
      RUN_OP_API_SYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor, stream_ptr);       \
    }                                                                                                                 \
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

class AclnnKernelMod : public KernelMod {
 public:
  explicit AclnnKernelMod(std::string &&op_type) : op_type_(std::move(op_type)) {}
  ~AclnnKernelMod() = default;

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

  template <typename... Args>
  void ParseGenExecutor(const std::tuple<Args...> &args) {
    if (is_dynamic_) {
      workspace_size_list_.clear();
      size_t size = std::get<0>(args);
      if (size != 0) {
        (void)workspace_size_list_.emplace_back(size);
      }
    }

    executor_ = std::get<1>(args);
    if (executor_ == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Please check op api's generate!";
    }
    release_func_ = std::get<2>(args);

    constexpr size_t kBoostGeneratorSize = 5;
    if constexpr (std::tuple_size_v<std::tuple<Args...>> == kBoostGeneratorSize) {
      hash_id_ = std::get<kHashIdIndex>(args);
      if (cache_hash_.count(hash_id_) != 0) {
        return;
      }
      constexpr size_t kHitIndex = 4;
      if (std::get<kHitIndex>(args)) {
        cache_hash_.insert(hash_id_);
      }
    }
  }

  void SetDynamic(bool is_dynamic) {
    std::lock_guard<std::mutex> lock(mtx_);
    is_dynamic_ = is_dynamic;
  }

 protected:
  template <size_t N, std::size_t... Is>
  auto GetTupleFrontImpl(const std::vector<KernelTensor *> &vecs, std::index_sequence<Is...>) {
    return std::make_tuple(vecs[Is]...);
  }

  template <size_t N>
  auto GetTupleFront(const std::vector<KernelTensor *> &vecs) {
    return GetTupleFrontImpl<N>(vecs, std::make_index_sequence<N>());
  }

  template <typename T, typename... Vecs>
  std::vector<T> ConcatVecs(const std::vector<T> &vec, const Vecs &... vecs) {
    std::vector<T> result = vec;
    (result.insert(result.end(), vecs.begin(), vecs.end()), ...);
    return result;
  }

  template <typename T, typename... Vecs>
  std::vector<T> ConcatVecs(const Vecs &... vecs) {
    static_assert((std::is_same_v<T, typename Vecs::value_type> && ...), "All vectors must have the same type!");
    std::vector<T> result;
    (result.insert(result.end(), vecs.begin(), vecs.end()), ...);
    return result;
  }

  template <size_t N, typename... Ts>
  auto GetKernelTuple(const std::vector<Ts> &... vecs) {
    const auto &new_vec = ConcatVecs(vecs...);
    if (new_vec.size() != N) {
      MS_LOG(EXCEPTION) << op_type_ << "'s config op input and output's size must be same, but get " << N << " with "
                        << new_vec.size();
    }
    const auto &result = GetTupleFront<N>(new_vec);
    return result;
  }

  aclOpExecutor *executor_{nullptr};
  CallBackFunc release_func_{nullptr};
  std::string op_type_;
  uint64_t hash_id_{0};
  std::unordered_set<uint64_t> cache_hash_;
  static bool is_dynamic_;
  std::mutex mtx_;
  std::unordered_map<uint64_t, std::list<CacheTuple>::iterator> hash_map_;
  std::list<CacheTuple> hash_cache_;
  size_t capacity_{128};

  static constexpr size_t kWsSizeIndex = 0;
  static constexpr size_t kHashIdIndex = 3;
};

using AclnnKernelModPtr = std::shared_ptr<AclnnKernelMod>;
using AclnnKernelModPtrList = std::vector<AclnnKernelModPtr>;

#define REGISTER_ACLNN_CLASS(TYPE)                                                                                    \
  template <size_t N>                                                                                                 \
  class Aclnn##TYPE##KernelMod : public AclnnKernelMod {                                                              \
   public:                                                                                                            \
    explicit Aclnn##TYPE##KernelMod(std::string &&op_type) : AclnnKernelMod(std::move(op_type)) {}                    \
    ~Aclnn##TYPE##KernelMod() = default;                                                                              \
    void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,                                                  \
                          const std::vector<KernelTensor *> &outputs) override {                                      \
      const auto &res_tuple = this->GetKernelTuple<N>(inputs, outputs);                                               \
      std::apply([this](const auto &... args) { GetWorkspaceForResize(args...); }, res_tuple);                        \
    }                                                                                                                 \
    bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,              \
                const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {                              \
      CallRun(stream_ptr, workspace, inputs, outputs);                                                                \
      return true;                                                                                                    \
    }                                                                                                                 \
                                                                                                                      \
   private:                                                                                                           \
    template <typename... Ts>                                                                                         \
    void CallRun(void *stream_ptr, const std::vector<KernelTensor *> &workspace, const std::vector<Ts> &... vecs) {   \
      const auto &res_tuple = this->GetKernelTuple<N>(vecs...);                                                       \
      std::apply(                                                                                                     \
        [this, stream_ptr, &workspace](const auto &... args) { return this->RunOp(stream_ptr, workspace, args...); }, \
        res_tuple);                                                                                                   \
    }                                                                                                                 \
                                                                                                                      \
    DEFINE_GET_WORKSPACE_FOR_RESIZE()                                                                                 \
  };

#define MS_ACLNN_KERNEL_FACTORY_REG(NAME, DERIVE_CLASS) MS_KERNEL_FACTORY_REG(AclnnKernelMod, NAME, DERIVE_CLASS)
#define MS_ACLNN_COMMON_KERNEL_FACTORY_REG(NAME, TYPE, N)                     \
  REGISTER_ACLNN_CLASS(NAME)                                                  \
  static const KernelRegistrar<AclnnKernelMod> g_##NAME##_AclnnKernelMod_reg( \
    #NAME, []() { return std::make_shared<Aclnn##NAME##KernelMod<N>>(#TYPE); });
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_KERNEL_MOD_H_
