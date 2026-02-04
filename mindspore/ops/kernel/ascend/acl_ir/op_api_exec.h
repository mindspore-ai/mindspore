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
#include <set>
#include "acl/acl_base.h"
#include "acl/acl.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "kernel/ascend/acl_ir/op_api_cache.h"
#include "kernel/ascend/acl_ir/op_api_util.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "kernel/ascend/visible.h"

namespace mindspore::device::ascend {
enum class ProcessCacheType {
  kReleaseParams,
  kReleaseParamsAndExecutor,
  kGetOutputShape,
  kUpdateTensorAddress,
};

using InitHugeMemThreadLocal = std::function<int(void *, bool)>;
using UnInitHugeMemThreadLocal = std::function<void(void *, bool)>;
using ReleaseHugeMem = std::function<void(void *, bool)>;
using ReleaseExecutor = std::function<int(device::ascend::aclOpExecutor *)>;
using ReleaseCallBack = std::function<void()>;
using ProcessCache = std::function<std::vector<ShapeVector>(const device::ascend::ProcessCacheType &,
                                                            const std::vector<std::vector<void *>> &)>;
using RunApiFunc = int (*)(void *, uint64_t, device::ascend::aclOpExecutor *, const aclrtStream);

extern std::set<std::string> sync_launch_api;

#define REGISTER_SYNC_OP(op_api)                              \
  do {                                                        \
    if (device::ascend::sync_launch_api.count(op_api) == 0) { \
      device::ascend::sync_launch_api.insert(op_api);         \
    }                                                         \
  } while (false)

class OPS_ASCEND_API OpApiDefaultResource {
 public:
  static OpApiDefaultResource &GetInstance();

  InitHugeMemThreadLocal init_mem_func();
  UnInitHugeMemThreadLocal uninit_mem_func();
  ReleaseHugeMem release_mem_func();
  ReleaseExecutor release_executor_func();

 private:
  OpApiDefaultResource() = default;
  ~OpApiDefaultResource() = default;

  InitHugeMemThreadLocal init_mem_func_{nullptr};
  UnInitHugeMemThreadLocal uninit_mem_func_{nullptr};
  ReleaseHugeMem release_mem_func_{nullptr};
  ReleaseExecutor release_executor_func_{nullptr};
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
      auto release_mem_func = device::ascend::OpApiDefaultResource::GetInstance().release_mem_func();
      if (release_mem_func) {
        release_mem_func(nullptr, false);
      }
      auto uninit_mem_func = device::ascend::OpApiDefaultResource::GetInstance().uninit_mem_func();
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

OPS_ASCEND_API void LoadOpApiLib();
OPS_ASCEND_API void AclnnInit();
OPS_ASCEND_API void AclnnFinalize();

template <typename T>
class GraphCache {
 public:
  explicit GraphCache(device::ascend::aclOpExecutor *executor, T &&param)
      : executor_(executor), converted_params_(param) {}
  std::vector<ShapeVector> operator()(const device::ascend::ProcessCacheType &process_cache_type,
                                      const std::vector<std::vector<void *>> &address_list = {}) {
    static auto release_executor_func = device::ascend::OpApiDefaultResource::GetInstance().release_executor_func();
    switch (process_cache_type) {
      case ProcessCacheType::kGetOutputShape:
        return FillShapeListFromTuple(converted_params_);
      case ProcessCacheType::kReleaseParamsAndExecutor:
        if (release_executor_func != nullptr) {
          release_executor_func(executor_);
        }
        ReleaseConvertTypes(converted_params_);
        break;
      case ProcessCacheType::kReleaseParams:
        ReleaseConvertTypes(converted_params_);
        break;
      case ProcessCacheType::kUpdateTensorAddress:
        UpdateAddressForTensor(executor_, address_list, converted_params_);
        break;
      default:
        break;
    }
    return {};
  }

 private:
  device::ascend::aclOpExecutor *executor_;
  T converted_params_;
};

template <typename T>
class ReleaseCall {
 public:
  explicit ReleaseCall(T &&param) : converted_params_(param) {}
  void operator()() {
    ReleaseConvertTypes(converted_params_);
    auto release_mem_func = device::ascend::OpApiDefaultResource::GetInstance().release_mem_func();
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

// check and throw only when enable uce.
#define CHECK_AND_THROW_RECOVERABLE_ERROR(aclnn_api)                                                          \
  do {                                                                                                        \
    static auto fail_cb =                                                                                     \
      GET_COMMON_CALLBACK(RunFailCallback, void, const char *, int, const char *, const std::string &, bool); \
    if (fail_cb != nullptr) {                                                                                 \
      fail_cb(FILE_NAME, __LINE__, __FUNCTION__, aclnn_api, true);                                            \
    }                                                                                                         \
  } while (false)

// For custom op generate executor.
#define GEN_CUSTOM_EXECUTOR(aclnn_api, ...)                                                                            \
  [](const std::string &api_str, const std::string &workspace_api_name, const auto &...args) -> auto {                 \
    static device::ascend::ApiCachePool api_cache_pool;                                                                \
    const char *api_name = api_cache_pool.get(api_str);                                                                \
    const auto get_workspace_size_func_ptr = device::ascend::GetOpApiFunc(workspace_api_name.c_str());                 \
    if (get_workspace_size_func_ptr == nullptr) {                                                                      \
      MS_LOG(EXCEPTION) << workspace_api_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!"; \
    }                                                                                                                  \
    uint64_t workspace_size = 0;                                                                                       \
    device::ascend::aclOpExecutor *executor = nullptr;                                                                 \
    std::function<void()> release_func = nullptr;                                                                      \
    uint64_t *workspace_size_addr = &workspace_size;                                                                   \
    device::ascend::aclOpExecutor **executor_addr = &executor;                                                         \
    auto process_cache = device::ascend::ProcessCache(nullptr);                                                        \
    if (HitCache(api_name, executor_addr, workspace_size_addr, args...)) {                                             \
      MS_LOG(DEBUG) << "gen executor aclnn cache hit.";                                                                \
      return std::make_tuple(workspace_size, executor, process_cache, release_func);                                   \
    }                                                                                                                  \
    MS_LOG(DEBUG) << "gen executor aclnn cache miss.";                                                                 \
    auto init_mem_func = device::ascend::OpApiDefaultResource::GetInstance().init_mem_func();                          \
    if (init_mem_func) {                                                                                               \
      init_mem_func(nullptr, false);                                                                                   \
    }                                                                                                                  \
    auto converted_params = device::ascend::ConvertTypes(args..., workspace_size_addr, executor_addr);                 \
    auto get_workspace_size_func = device::ascend::ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);  \
    auto workspace_status = device::ascend::call(get_workspace_size_func, converted_params);                           \
    if (workspace_status != 0) {                                                                                       \
      MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";                                        \
    }                                                                                                                  \
    auto releas_call = device::ascend::ReleaseCall(std::move(converted_params));                                       \
    release_func = std::function<void()>(releas_call);                                                                 \
    auto graph_cache = device::ascend::GraphCache(executor, std::move(converted_params));                              \
    process_cache = device::ascend::ProcessCache(graph_cache);                                                         \
    auto uninit_mem_func = device::ascend::OpApiDefaultResource::GetInstance().uninit_mem_func();                      \
    if (uninit_mem_func) {                                                                                             \
      uninit_mem_func(nullptr, false);                                                                                 \
    }                                                                                                                  \
    device::ascend::UninitCacheThreadLocal();                                                                          \
    return std::make_tuple(workspace_size, executor, process_cache, release_func);                                     \
  }(aclnn_api, aclnn_api + "GetWorkspaceSize", __VA_ARGS__)

// For generate executor.
#define GEN_EXECUTOR(aclnn_api, ...)                                                                              \
  [](const std::string &api_str, const std::string &workspace_api_name, const auto &...args) -> auto {            \
    static mindspore::device::ascend::ApiCachePool api_cache_pool;                                                \
    const char *api_name = api_cache_pool.get(api_str);                                                           \
    static const auto get_workspace_size_func_ptr =                                                               \
      mindspore::device::ascend::GetOpApiFunc(workspace_api_name.c_str());                                        \
    if (get_workspace_size_func_ptr == nullptr) {                                                                 \
      MS_LOG(EXCEPTION) << workspace_api_name << " not in " << mindspore::device::ascend::GetOpApiLibName()       \
                        << ", please check!";                                                                     \
    }                                                                                                             \
    uint64_t workspace_size = 0;                                                                                  \
    mindspore::device::ascend::aclOpExecutor *executor = nullptr;                                                 \
    std::function<void()> release_func = nullptr;                                                                 \
    uint64_t *workspace_size_addr = &workspace_size;                                                              \
    mindspore::device::ascend::aclOpExecutor **executor_addr = &executor;                                         \
    auto process_cache = mindspore::device::ascend::ProcessCache(nullptr);                                        \
    if (mindspore::device::ascend::sync_launch_api.count(std::string(api_name)) == 0 &&                           \
        mindspore::device::ascend::HitCache(api_name, executor_addr, workspace_size_addr, args...)) {             \
      MS_VLOG(mindspore::VLogLevel::VL_ACLNN_OP) << api_name << " gen executor hit cache.";                       \
      return std::make_tuple(workspace_size, executor, process_cache, release_func);                              \
    }                                                                                                             \
    MS_VLOG(mindspore::VLogLevel::VL_ACLNN_OP) << api_name << " gen executor miss cache.";                        \
    auto init_mem_func = mindspore::device::ascend::OpApiDefaultResource::GetInstance().init_mem_func();          \
    if (init_mem_func) {                                                                                          \
      init_mem_func(nullptr, false);                                                                              \
    }                                                                                                             \
    auto converted_params = mindspore::device::ascend::ConvertTypes(args..., workspace_size_addr, executor_addr); \
    static auto get_workspace_size_func =                                                                         \
      mindspore::device::ascend::ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);               \
    auto workspace_status = mindspore::device::ascend::call(get_workspace_size_func, converted_params);           \
    if (workspace_status != 0) {                                                                                  \
      MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";                                   \
    }                                                                                                             \
    auto releas_call = mindspore::device::ascend::ReleaseCall(std::move(converted_params));                       \
    release_func = std::function<void()>(releas_call);                                                            \
    auto graph_cache = mindspore::device::ascend::GraphCache(executor, std::move(converted_params));              \
    process_cache = mindspore::device::ascend::ProcessCache(graph_cache);                                         \
    auto uninit_mem_func = mindspore::device::ascend::OpApiDefaultResource::GetInstance().uninit_mem_func();      \
    if (uninit_mem_func) {                                                                                        \
      uninit_mem_func(nullptr, false);                                                                            \
    }                                                                                                             \
    mindspore::device::ascend::UninitCacheThreadLocal();                                                          \
    return std::make_tuple(workspace_size, executor, process_cache, release_func);                                \
  }(aclnn_api, aclnn_api + "GetWorkspaceSize", __VA_ARGS__)

// For generate executor without cache.
#define GEN_EXECUTOR_CUST(aclnn_api, ...)                                                                              \
  [](const std::string &workspace_api_name, auto &...args) -> auto {                                                   \
    static const auto get_workspace_size_func_ptr = device::ascend::GetOpApiFunc(workspace_api_name.c_str());          \
    if (get_workspace_size_func_ptr == nullptr) {                                                                      \
      MS_LOG(EXCEPTION) << workspace_api_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!"; \
    }                                                                                                                  \
    uint64_t workspace_size = 0;                                                                                       \
    uint64_t *workspace_size_addr = &workspace_size;                                                                   \
    device::ascend::aclOpExecutor *executor = nullptr;                                                                 \
    device::ascend::aclOpExecutor **executor_addr = &executor;                                                         \
    auto convert_params = device::ascend::ConvertTypes(args..., workspace_size_addr, executor_addr);                   \
    static auto get_workspace_size_func =                                                                              \
      device::ascend::ConvertToOpApiFunc(convert_params, get_workspace_size_func_ptr);                                 \
    auto workspace_status = device::ascend::call(get_workspace_size_func, convert_params);                             \
    if (workspace_status != 0) {                                                                                       \
      MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";                                        \
    }                                                                                                                  \
    return std::make_tuple(workspace_size, executor,                                                                   \
                           device::ascend::OpApiParams<decltype(convert_params)>(std::move(convert_params)));          \
  }(aclnn_api + "GetWorkspaceSize", __VA_ARGS__)

// For speed up generate executor without hash_id.
#define GEN_EXECUTOR_BOOST(aclnn_api, hash_id, ...)                                                                    \
  [](const std::string &api_str, const std::string &workspace_api_name, uint64_t hash_id,                              \
     const auto &...args) -> auto {                                                                                    \
    static device::ascend::ApiCachePool api_cache_pool;                                                                \
    const char *api_name = api_cache_pool.get(api_str);                                                                \
    static const auto get_workspace_size_func_ptr = device::ascend::GetOpApiFunc(workspace_api_name.c_str());          \
    if (get_workspace_size_func_ptr == nullptr) {                                                                      \
      MS_LOG(EXCEPTION) << workspace_api_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!"; \
    }                                                                                                                  \
    uint64_t workspace_size = 0;                                                                                       \
    device::ascend::aclOpExecutor *executor = nullptr;                                                                 \
    std::function<void()> release_func = nullptr;                                                                      \
    uint64_t *workspace_size_addr = &workspace_size;                                                                   \
    device::ascend::aclOpExecutor **executor_addr = &executor;                                                         \
    uint64_t new_hash_id = hash_id;                                                                                    \
    if (HitCacheSingle(api_name, executor_addr, workspace_size_addr, &new_hash_id, args...)) {                         \
      MS_VLOG(VL_ACLNN_OP) << api_name << " gen executor hit cache, hash id: " << new_hash_id;                         \
      return std::make_tuple(workspace_size, executor, release_func, new_hash_id, true);                               \
    }                                                                                                                  \
    MS_VLOG(VL_ACLNN_OP) << api_name << " gen executor miss cache, hash id: " << new_hash_id;                          \
    auto init_mem_func = device::ascend::OpApiDefaultResource::GetInstance().init_mem_func();                          \
    if (init_mem_func) {                                                                                               \
      init_mem_func(nullptr, false);                                                                                   \
    }                                                                                                                  \
    auto converted_params = device::ascend::ConvertTypes(args..., workspace_size_addr, executor_addr);                 \
    static auto get_workspace_size_func =                                                                              \
      device::ascend::ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);                               \
    auto workspace_status = device::ascend::call(get_workspace_size_func, converted_params);                           \
    if (workspace_status != 0) {                                                                                       \
      MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";                                        \
    }                                                                                                                  \
    auto releas_call = device::ascend::ReleaseCall(std::move(converted_params));                                       \
    release_func = std::function<void()>(releas_call);                                                                 \
    auto uninit_mem_func = device::ascend::OpApiDefaultResource::GetInstance().uninit_mem_func();                      \
    if (uninit_mem_func) {                                                                                             \
      uninit_mem_func(nullptr, false);                                                                                 \
    }                                                                                                                  \
    device::ascend::UninitCacheThreadLocal();                                                                          \
    return std::make_tuple(workspace_size, executor, release_func, new_hash_id, false);                                \
  }(aclnn_api, aclnn_api + "GetWorkspaceSize", hash_id, __VA_ARGS__)

// Gen acltensor for view op
#define GEN_EXECUTOR_FOR_VIEW(aclnn_api, ...)                                                          \
  [](const std::string &workspace_api_name, const auto &...args) -> auto {                             \
    uint64_t workspace_size = 0;                                                                       \
    device::ascend::aclOpExecutor *executor = nullptr;                                                 \
    uint64_t *workspace_size_addr = &workspace_size;                                                   \
    device::ascend::aclOpExecutor **executor_addr = &executor;                                         \
    auto converted_params = device::ascend::ConvertTypes(args..., workspace_size_addr, executor_addr); \
    auto graph_cache = device::ascend::GraphCache(executor, std::move(converted_params));              \
    auto process_cache = device::ascend::ProcessCache(graph_cache);                                    \
  }

// First stage for static graph.
#define GEN_EXECUTOR_FOR_RESIZE(aclnn_api, ...)                                                                   \
  [](const std::string &workspace_api_name, const auto &...args) -> auto {                                        \
    static const auto get_workspace_size_func_ptr =                                                               \
      mindspore::device::ascend::GetOpApiFunc(workspace_api_name.c_str());                                        \
    if (get_workspace_size_func_ptr == nullptr) {                                                                 \
      MS_LOG(EXCEPTION) << workspace_api_name << " not in " << mindspore::device::ascend::GetOpApiLibName()       \
                        << ", please check!";                                                                     \
    }                                                                                                             \
    uint64_t workspace_size = 0;                                                                                  \
    mindspore::device::ascend::aclOpExecutor *executor = nullptr;                                                 \
    uint64_t *workspace_size_addr = &workspace_size;                                                              \
    mindspore::device::ascend::aclOpExecutor **executor_addr = &executor;                                         \
    auto converted_params = mindspore::device::ascend::ConvertTypes(args..., workspace_size_addr, executor_addr); \
    static auto get_workspace_size_func =                                                                         \
      mindspore::device::ascend::ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);               \
    auto workspace_status = mindspore::device::ascend::call(get_workspace_size_func, converted_params);           \
    if (workspace_status != 0) {                                                                                  \
      CHECK_AND_THROW_RECOVERABLE_ERROR(workspace_api_name);                                                      \
      MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";                                   \
    }                                                                                                             \
    int32_t repeat_ret = mindspore::device::ascend::SetExecutorRepeatable(workspace_api_name, executor);          \
    auto graph_cache = mindspore::device::ascend::GraphCache(executor, std::move(converted_params));              \
    auto process_cache = mindspore::device::ascend::ProcessCache(graph_cache);                                    \
    return std::make_tuple(workspace_size, executor, process_cache, repeat_ret);                                  \
  }(aclnn_api + "GetWorkspaceSize", __VA_ARGS__)

#define GEN_CUSTOM_EXECUTOR_FOR_RESIZE(aclnn_api, ...)                                                                 \
  [](const std::string &workspace_api_name, const auto &...args) -> auto {                                             \
    const auto get_workspace_size_func_ptr = device::ascend::GetOpApiFunc(workspace_api_name.c_str());                 \
    if (get_workspace_size_func_ptr == nullptr) {                                                                      \
      MS_LOG(EXCEPTION) << workspace_api_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!"; \
    }                                                                                                                  \
    uint64_t workspace_size = 0;                                                                                       \
    device::ascend::aclOpExecutor *executor = nullptr;                                                                 \
    uint64_t *workspace_size_addr = &workspace_size;                                                                   \
    device::ascend::aclOpExecutor **executor_addr = &executor;                                                         \
    auto converted_params = device::ascend::ConvertTypes(args..., workspace_size_addr, executor_addr);                 \
    auto get_workspace_size_func = device::ascend::ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);  \
    auto workspace_status = device::ascend::call(get_workspace_size_func, converted_params);                           \
    if (workspace_status != 0) {                                                                                       \
      CHECK_AND_THROW_RECOVERABLE_ERROR(workspace_api_name);                                                           \
      MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";                                        \
    }                                                                                                                  \
    int32_t repeat_ret = device::ascend::SetExecutorRepeatable(workspace_api_name, executor);                          \
    auto graph_cache = device::ascend::GraphCache(executor, std::move(converted_params));                              \
    auto process_cache = device::ascend::ProcessCache(graph_cache);                                                    \
    return std::make_tuple(workspace_size, executor, process_cache, repeat_ret);                                       \
  }(aclnn_api + "GetWorkspaceSize", __VA_ARGS__)

// Update tensor for static graph.
#define UPDATE_TENSOR_FOR_LAUNCH(process_cache, ...)                                     \
  do {                                                                                   \
    const auto &address_list = device::ascend::GetTensorAddress(__VA_ARGS__);            \
    process_cache(device::ascend::ProcessCacheType::kUpdateTensorAddress, address_list); \
  } while (false)

// Async run op.
#define RUN_OP_API_ASYNC(aclnn_api, workspace_addr, workspace_size, executor, acl_stream, release_func)       \
  do {                                                                                                        \
    static const auto op_api_func = mindspore::device::ascend::GetOpApiFunc(aclnn_api.c_str());               \
    if (op_api_func == nullptr) {                                                                             \
      MS_LOG(EXCEPTION) << aclnn_api << " not in " << mindspore::device::ascend::GetOpApiLibName()            \
                        << ", please check!";                                                                 \
    }                                                                                                         \
    auto run_api_func = reinterpret_cast<mindspore::device::ascend::RunApiFunc>(op_api_func);                 \
    auto api_ret = run_api_func(workspace_addr, workspace_size, executor, acl_stream);                        \
    if (api_ret != 0) {                                                                                       \
      CHECK_AND_THROW_RECOVERABLE_ERROR(aclnn_api);                                                           \
      MS_LOG(EXCEPTION) << "Call " << aclnn_api << " failed, detail:" << CALL_ASCEND_API(aclGetRecentErrMsg); \
    }                                                                                                         \
    if (release_func != nullptr) {                                                                            \
      release_func();                                                                                         \
    }                                                                                                         \
  } while (false)

// Async run custom op.
#define RUN_CUSTOM_OP_API_ASYNC(aclnn_api, workspace_addr, workspace_size, executor, acl_stream, release_func) \
  do {                                                                                                         \
    const auto op_api_func = device::ascend::GetOpApiFunc(aclnn_api.c_str());                                  \
    if (MS_UNLIKELY(op_api_func == nullptr)) {                                                                 \
      MS_LOG(EXCEPTION) << aclnn_api << " not in " << device::ascend::GetOpApiLibName() << ", please check!";  \
    }                                                                                                          \
    static auto run_api_func = reinterpret_cast<device::ascend::RunApiFunc>(op_api_func);                      \
    auto api_ret = run_api_func(workspace_addr, workspace_size, executor, acl_stream);                         \
    if (MS_UNLIKELY(api_ret != 0)) {                                                                           \
      CHECK_AND_THROW_RECOVERABLE_ERROR(aclnn_api);                                                            \
      MS_LOG(EXCEPTION) << "Call " << aclnn_api << " failed, detail:" << CALL_ASCEND_API(aclGetRecentErrMsg);  \
    }                                                                                                          \
    if (release_func != nullptr) {                                                                             \
      release_func();                                                                                          \
    }                                                                                                          \
  } while (false)

// Sync run op.
#define RUN_OP_API_SYNC(aclnn_api, workspace_addr, workspace_size, executor, acl_stream)                      \
  do {                                                                                                        \
    static const auto op_api_func = device::ascend::GetOpApiFunc(aclnn_api.c_str());                          \
    if (MS_UNLIKELY(op_api_func == nullptr)) {                                                                \
      MS_LOG(EXCEPTION) << aclnn_api << " not in " << device::ascend::GetOpApiLibName() << ", please check!"; \
    }                                                                                                         \
    static auto run_api_func = reinterpret_cast<device::ascend::RunApiFunc>(op_api_func);                     \
    auto api_ret = run_api_func(workspace_addr, workspace_size, executor, acl_stream);                        \
    if (MS_UNLIKELY(api_ret != 0)) {                                                                          \
      CHECK_AND_THROW_RECOVERABLE_ERROR(aclnn_api);                                                           \
      MS_LOG(EXCEPTION) << "Call " << aclnn_api << " failed, detail:" << CALL_ASCEND_API(aclGetRecentErrMsg); \
    }                                                                                                         \
  } while (false)
}  // namespace  mindspore::device::ascend

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_EXEC_H_
