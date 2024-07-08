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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_EXEC_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_EXEC_H_

#include <dlfcn.h>
#include <vector>
#include <functional>
#include <string>
#include <utility>
#include <unordered_map>
#include "acl/acl_base.h"
#include "acl/acl.h"
#include "transform/acl_ir/op_api_convert.h"
#include "transform/acl_ir/op_api_cache.h"
#include "transform/acl_ir/op_api_util.h"
#include "transform/acl_ir/acl_allocator.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/acl_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace transform {
using InitHugeMemThreadLocal = std::function<int(void *, bool)>;
using UnInitHugeMemThreadLocal = std::function<void(void *, bool)>;
using ReleaseHugeMem = std::function<void(void *, bool)>;
using ReleaseCallBack = std::function<void()>;
using RunApiFunc = int (*)(void *, uint64_t, transform::aclOpExecutor *, const aclrtStream);

class OpApiDefaultResource {
 public:
  static OpApiDefaultResource &GetInstance();

  InitHugeMemThreadLocal init_mem_func();
  UnInitHugeMemThreadLocal uninit_mem_func();
  ReleaseHugeMem release_mem_func();

 private:
  OpApiDefaultResource() = default;
  ~OpApiDefaultResource() = default;

  InitHugeMemThreadLocal init_mem_func_{nullptr};
  UnInitHugeMemThreadLocal uninit_mem_func_{nullptr};
  ReleaseHugeMem release_mem_func_{nullptr};
};

template <typename Tuple>
class OpApiParams {
 public:
  explicit OpApiParams(Tuple &&converted_params) : converted_params_(std::move(converted_params)) {}
  explicit OpApiParams(Tuple &&converted_params, bool mem_clear)
      : converted_params_(std::move(converted_params)), mem_clear_(mem_clear) {}
  explicit OpApiParams(OpApiParams &&other) : converted_params_(std::move(other.converted_params_)) {
    need_free_ = other.need_free_;
    mem_clear_ = other.mem_clear_;
    other.need_free_ = false;
    other.mem_clear_ = false;
  }
  OpApiParams &operator=(OpApiParams &&other) {
    if (this == &other) {
      return *this;
    }

    if (need_free_) {
      ReleaseConvertTypes(converted_params_);
    }

    converted_params_ = std::move(other.converted_params_);
    need_free_ = other.need_free_;
    mem_clear_ = other.mem_clear_;
    other.need_free_ = false;
    other.mem_clear_ = false;
    return *this;
  }

  OpApiParams() = delete;
  OpApiParams(const OpApiParams &other) = delete;
  OpApiParams &operator=(const OpApiParams &other) = delete;

  ~OpApiParams() {
    if (need_free_) {
      ReleaseConvertTypes(converted_params_);
    }
    if (mem_clear_) {
      auto release_mem_func = transform::OpApiDefaultResource::GetInstance().release_mem_func();
      if (release_mem_func) {
        release_mem_func(nullptr, false);
      }
      auto uninit_mem_func = transform::OpApiDefaultResource::GetInstance().uninit_mem_func();
      if (uninit_mem_func) {
        uninit_mem_func(nullptr, false);
      }
    }
  }

  const Tuple &converted_params() const { return converted_params_; }

  template <size_t i>
  auto get() {
    return std::get<i>(converted_params_);
  }

 private:
  Tuple converted_params_;
  bool need_free_{true};
  bool mem_clear_{false};
};

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

// Get output shape from acl tensor.
ShapeVector UpdateOutputShape(const aclTensor *tensor);
void LoadOpApiLib();
void AclnnInit();
void AclnnFinalize();

template <typename T>
class ReleaseCall {
 public:
  explicit ReleaseCall(T &&param) : converted_params_(param) {}
  void operator()() {
    ReleaseConvertTypes(converted_params_);
    auto release_mem_func = transform::OpApiDefaultResource::GetInstance().release_mem_func();
    if (release_mem_func) {
      release_mem_func(nullptr, false);
    }
  }

 private:
  T converted_params_;
};

class ApiCachePool {
 public:
  ApiCachePool() = default;
  ~ApiCachePool() = default;

  const char *get(const std::string &str) {
    auto it = pool_.find(str);
    if (it != pool_.end()) {
      return it->second.c_str();
    }
    auto [map_iter, inserted] = pool_.emplace(str, str);
    if (!inserted) {
      MS_LOG(EXCEPTION) << "Failed to cache api.";
    }
    return map_iter->second.c_str();
  }

 private:
  std::unordered_map<std::string, std::string> pool_;
};

// For normal generate executor.
#define GEN_EXECUTOR(aclnn_api, ...)                                                                              \
  [](const std::string &api_str, const std::string &workspace_api_name, const auto &... args) -> auto {           \
    static transform::ApiCachePool api_cache_pool;                                                                \
    const char *api_name = api_cache_pool.get(api_str);                                                           \
    static const auto get_workspace_size_func_ptr = transform::GetOpApiFunc(workspace_api_name.c_str());          \
    if (get_workspace_size_func_ptr == nullptr) {                                                                 \
      MS_LOG(EXCEPTION) << workspace_api_name << " not in " << transform::GetOpApiLibName() << ", please check!"; \
    }                                                                                                             \
    uint64_t workspace_size = 0;                                                                                  \
    transform::aclOpExecutor *executor = nullptr;                                                                 \
    std::function<void()> release_func = nullptr;                                                                 \
    uint64_t *workspace_size_addr = &workspace_size;                                                              \
    transform::aclOpExecutor **executor_addr = &executor;                                                         \
    if (HitCache(api_name, executor_addr, workspace_size_addr, args...)) {                                        \
      MS_LOG(DEBUG) << "gen executor aclnn cache hit.";                                                                                                            \
      return std::make_tuple(workspace_size, executor, release_func);                                             \
    }                                                                                                             \
    MS_LOG(DEBUG) << "gen executor aclnn cache miss.";                                                                                                              \
    auto init_mem_func = transform::OpApiDefaultResource::GetInstance().init_mem_func();                          \
    if (init_mem_func) {                                                                                          \
      init_mem_func(nullptr, false);                                                                              \
    }                                                                                                             \
    auto converted_params = transform::ConvertTypes(args..., workspace_size_addr, executor_addr);                 \
    static auto get_workspace_size_func =                                                                         \
      transform::ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);                               \
    auto workspace_status = transform::call(get_workspace_size_func, converted_params);                           \
    if (workspace_status != 0) {                                                                                  \
      MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";                                   \
    }                                                                                                             \
    auto releas_call = transform::ReleaseCall(std::move(converted_params));                                       \
    release_func = std::function<void()>(releas_call);                                                            \
    auto uninit_mem_func = transform::OpApiDefaultResource::GetInstance().uninit_mem_func();                      \
    if (uninit_mem_func) {                                                                                        \
      uninit_mem_func(nullptr, false);                                                                            \
    }                                                                                                             \
    return std::make_tuple(workspace_size, executor, release_func);                                               \
  }                                                                                                               \
  (aclnn_api, aclnn_api + "GetWorkspaceSize", __VA_ARGS__)

// For custom generate executor.
#define GEN_EXECUTOR_CUST(aclnn_api, use_huge_pages, ...)                                                         \
  [](const std::string &workspace_api_name, bool use_huge_pages, auto &... args) -> auto {                        \
    static const auto get_workspace_size_func_ptr = transform::GetOpApiFunc(workspace_api_name.c_str());          \
    if (get_workspace_size_func_ptr == nullptr) {                                                                 \
      MS_LOG(EXCEPTION) << workspace_api_name << " not in " << transform::GetOpApiLibName() << ", please check!"; \
    }                                                                                                             \
    static const auto init_cache_thread_local = transform::GetOpApiFunc("InitPTACacheThreadLocal");               \
    static const auto set_hash_key = transform::GetOpApiFunc("SetPTAHashKey");                                    \
    transform::InitCacheThreadLocal init_cache_thread_local_func =                                                \
      reinterpret_cast<transform::InitCacheThreadLocal>(init_cache_thread_local);                                 \
    transform::SetHashKey set_hash_key_func = reinterpret_cast<transform::SetHashKey>(set_hash_key);              \
    if (init_cache_thread_local_func && set_hash_key_func) {                                                      \
      init_cache_thread_local_func();                                                                             \
      set_hash_key_func(0);                                                                                       \
    }                                                                                                             \
    uint64_t workspace_size = 0;                                                                                  \
    uint64_t *workspace_size_addr = &workspace_size;                                                              \
    transform::aclOpExecutor *executor = nullptr;                                                                 \
    transform::aclOpExecutor **executor_addr = &executor;                                                         \
    auto init_mem_func = transform::OpApiDefaultResource::GetInstance().init_mem_func();                          \
    if (use_huge_pages && init_mem_func) {                                                                        \
      init_mem_func(nullptr, false);                                                                              \
    }                                                                                                             \
    auto converted_params = transform::ConvertTypes(args..., workspace_size_addr, executor_addr);                 \
    static auto get_workspace_size_func =                                                                         \
      transform::ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);                               \
    auto workspace_status = transform::call(get_workspace_size_func, converted_params);                           \
    if (workspace_status != 0) {                                                                                  \
      MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";                                   \
    }                                                                                                             \
    return std::make_tuple(workspace_size, executor,                                                              \
                           transform::OpApiParams<decltype(converted_params)>(std::move(converted_params)),       \
                           use_huge_pages);                                                                       \
  }                                                                                                               \
  (aclnn_api + "GetWorkspaceSize", use_huge_pages, __VA_ARGS__)

// For speed up generate executor.
#define GEN_EXECUTOR_BOOST(aclnn_api, hash_id, ...)                                                               \
  [](const std::string &api_str, const std::string &workspace_api_name, uint64_t hash_id,                         \
     const auto &... args) -> auto {                                                                              \
    static transform::ApiCachePool api_cache_pool;                                                                \
    const char *api_name = api_cache_pool.get(api_str);                                                           \
    static const auto get_workspace_size_func_ptr = transform::GetOpApiFunc(workspace_api_name.c_str());          \
    if (get_workspace_size_func_ptr == nullptr) {                                                                 \
      MS_LOG(EXCEPTION) << workspace_api_name << " not in " << transform::GetOpApiLibName() << ", please check!"; \
    }                                                                                                             \
    uint64_t workspace_size = 0;                                                                                  \
    transform::aclOpExecutor *executor = nullptr;                                                                 \
    std::function<void()> release_func = nullptr;                                                                 \
    uint64_t *workspace_size_addr = &workspace_size;                                                              \
    transform::aclOpExecutor **executor_addr = &executor;                                                         \
    uint64_t new_hash_id = hash_id;                                                                               \
    if (HitCacheSingle(api_name, executor_addr, workspace_size_addr, &new_hash_id, args...)) {                    \
      return std::make_tuple(workspace_size, executor, release_func, new_hash_id, true);                          \
    }                                                                                                             \
    auto init_mem_func = transform::OpApiDefaultResource::GetInstance().init_mem_func();                          \
    if (init_mem_func) {                                                                                          \
      init_mem_func(nullptr, false);                                                                              \
    }                                                                                                             \
    auto converted_params = transform::ConvertTypes(args..., workspace_size_addr, executor_addr);                 \
    static auto get_workspace_size_func =                                                                         \
      transform::ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);                               \
    auto workspace_status = transform::call(get_workspace_size_func, converted_params);                           \
    if (workspace_status != 0) {                                                                                  \
      MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";                                   \
    }                                                                                                             \
    auto releas_call = transform::ReleaseCall(std::move(converted_params));                                       \
    release_func = std::function<void()>(releas_call);                                                            \
    auto uninit_mem_func = transform::OpApiDefaultResource::GetInstance().uninit_mem_func();                      \
    if (uninit_mem_func) {                                                                                        \
      uninit_mem_func(nullptr, false);                                                                            \
    }                                                                                                             \
    return std::make_tuple(workspace_size, executor, release_func, new_hash_id, false);                           \
  }                                                                                                               \
  (aclnn_api, aclnn_api + "GetWorkspaceSize", hash_id, __VA_ARGS__)

// Async run op.
#define RUN_OP_API_ASYNC(aclnn_api, workspace_addr, workspace_size, executor, acl_stream, release_func)       \
  do {                                                                                                        \
    static const auto op_api_func = transform::GetOpApiFunc(aclnn_api.c_str());                               \
    if (op_api_func == nullptr) {                                                                             \
      MS_LOG(EXCEPTION) << aclnn_api << " not in " << transform::GetOpApiLibName() << ", please check!";      \
    }                                                                                                         \
    auto run_api_func = reinterpret_cast<transform::RunApiFunc>(op_api_func);                                 \
    auto api_ret = run_api_func(workspace_addr, workspace_size, executor, acl_stream);                        \
    if (api_ret != 0) {                                                                                       \
      MS_LOG(EXCEPTION) << "Call " << aclnn_api << " failed, detail:" << CALL_ASCEND_API(aclGetRecentErrMsg); \
    }                                                                                                         \
    if (release_func != nullptr) {                                                                            \
      release_func();                                                                                         \
    }                                                                                                         \
  } while (false)

// Sync run op.
#define RUN_OP_API_SYNC(aclnn_api, workspace_addr, workspace_size, executor, acl_stream)                             \
  do {                                                                                                               \
    static const auto op_api_func = transform::GetOpApiFunc(aclnn_api.c_str());                                      \
    if (op_api_func == nullptr) {                                                                                    \
      MS_LOG(EXCEPTION) << aclnn_api << " not in " << transform::GetOpApiLibName() << ", please check!";             \
    }                                                                                                                \
    auto run_api_func = reinterpret_cast<transform::RunApiFunc>(op_api_func);                                        \
    auto api_ret = run_api_func(workspace_addr, workspace_size, executor, acl_stream);                               \
    if (api_ret != 0) {                                                                                              \
      MS_LOG(EXCEPTION) << "Call " << aclnn_api << " failed, detail:" << CALL_ASCEND_API(aclGetRecentErrMsg);        \
    }                                                                                                                \
    auto ret = CALL_ASCEND_API(aclrtSynchronizeStream, acl_stream);                                                  \
    if (ret != 0) {                                                                                                  \
      MS_LOG(EXCEPTION) << "Sync stream " << aclnn_api << " failed, detail:" << CALL_ASCEND_API(aclGetRecentErrMsg); \
    }                                                                                                                \
  } while (false)
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_EXEC_H_
