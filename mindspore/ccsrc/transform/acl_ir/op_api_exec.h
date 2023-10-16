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
#include "acl/acl_base.h"
#include "acl/acl.h"
#include "transform/acl_ir/op_api_convert.h"
#include "transform/acl_ir/acl_allocator.h"

namespace mindspore {
namespace transform {
using InitHugeMemThreadLocal = std::function<int(void *, bool)>;
using UnInitHugeMemThreadLocal = std::function<void(void *, bool)>;
using ReleaseHugeMem = std::function<void(void *, bool)>;
using ReleaseCallBack = std::function<void()>;

class OpApiDefaultResource {
 public:
  static OpApiDefaultResource &GetInstance();

  InitHugeMemThreadLocal init_mem_func();
  UnInitHugeMemThreadLocal uninit_mem_func();
  ReleaseHugeMem release_mem_func();

  void *AllocWorkspace(size_t size) { return allocator.AllocFunc(&allocator, size); }

  void FreeWorkspace(void *block) { allocator.FreeFunc(&allocator, block); }

 private:
  OpApiDefaultResource() { allocator.Initialize(); }
  ~OpApiDefaultResource() { allocator.Finalize(); }

  InitHugeMemThreadLocal init_mem_func_{nullptr};
  UnInitHugeMemThreadLocal uninit_mem_func_{nullptr};
  ReleaseHugeMem release_mem_func_{nullptr};

  AclAllocator allocator;
};

template <typename Tuple>
class OpApiParams {
 public:
  explicit OpApiParams(Tuple &&converted_params) : converted_params_(std::move(converted_params)) {}
  explicit OpApiParams(OpApiParams &&other) : converted_params_(std::move(other.converted_params_)) {
    other.need_free_ = false;
  }
  OpApiParams &operator=(OpApiParams &&other) {
    if (this == &other) {
      return *this;
    }

    if (need_free_) {
      ReleaseConvertTypes(converted_params_);
    }
    converted_params_ = std::move(other.converted_params_);
    need_free_ = true;
    other.need_free_ = false;
    return *this;
  }

  OpApiParams() = delete;
  OpApiParams(const OpApiParams &other) = delete;
  OpApiParams &operator=(const OpApiParams &other) = delete;

  ~OpApiParams() {
    if (need_free_) {
      ReleaseConvertTypes(converted_params_);
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

template <typename Tuple, size_t... I>
auto PackageParams(Tuple t, std::index_sequence<I...>) {
  return std::make_tuple(std::get<I>(t)...);
}

template <size_t N, typename Tuple>
auto PackageParams(Tuple t) {
  return PackageParams(t, std::make_index_sequence<N>{});
}

// Get Workspace size and op executor.
template <typename... Args>
auto GenerateExecutor(const std::string &aclnn_api, Args &... args) {
  auto workspace_func_name = aclnn_api + "GetWorkspaceSize";
  static const auto get_workspace_size_func_ptr = GetOpApiFunc(workspace_func_name.c_str());
  if (get_workspace_size_func_ptr == nullptr) {
    MS_LOG(EXCEPTION) << aclnn_api << " get_workspace_size func not in " << GetOpApiLibName() << ", please check!";
  }
  uint64_t workspace_size = 0;
  uint64_t *workspace_size_addr = &workspace_size;
  aclOpExecutor *executor = nullptr;
  aclOpExecutor **executor_addr = &executor;
  auto init_mem_func = OpApiDefaultResource::GetInstance().init_mem_func();
  if (init_mem_func) {
    init_mem_func(nullptr, false);
  }
  auto converted_params = ConvertTypes(args..., workspace_size_addr, executor_addr);
  static auto get_workspace_size_func = ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);
  auto workspace_status = call(get_workspace_size_func, converted_params);
  if (workspace_status != 0) {
    MS_LOG(EXCEPTION) << aclnn_api << " not in " << GetOpApiLibName() << ", please check!";
  }
  constexpr auto size = std::tuple_size<decltype(converted_params)>::value - 2;
  return std::make_tuple(PackageParams<size>(converted_params), workspace_size, executor);
}

void RunOpApi(const std::string &aclnn_api, const aclrtStream acl_stream, void *workspace_addr, uint64_t workspace_size,
              aclOpExecutor *executor, const ReleaseCallBack &release_func);

// Get output shape from acl tensor.
ShapeVector UpdateOutputShape(const aclTensor *tensor);

#define GEN_EXECUTOR(aclnn_api, ...)                                                                    \
  [](const std::string &api_name, auto &... args) -> auto {                                             \
    auto [converted_params, workspace_size, executor] = transform::GenerateExecutor(api_name, args...); \
    auto release_func = [converted_params]() -> void {                                                  \
      ReleaseConvertTypes(converted_params);                                                            \
      auto release_mem_func = transform::OpApiDefaultResource::GetInstance().release_mem_func();        \
      if (release_mem_func) {                                                                           \
        release_mem_func(nullptr, false);                                                               \
      }                                                                                                 \
      auto uninit_mem_func = transform::OpApiDefaultResource::GetInstance().uninit_mem_func();          \
      if (uninit_mem_func) {                                                                            \
        uninit_mem_func(nullptr, false);                                                                \
      }                                                                                                 \
    };                                                                                                  \
    return std::make_tuple(workspace_size, executor, release_func);                                     \
  }                                                                                                     \
  (#aclnn_api, __VA_ARGS__)

#define GEN_EXECUTOR_CUSTOM(aclnn_api, ...)                                                        \
  [](const std::string &api_name, auto &... args) -> auto {                                        \
    auto converted_params = transform::GenerateExecutor(api_name, args...);                        \
    auto real_params = std::get<0>(converted_params);                                              \
    return std::make_tuple(std::get<1>(converted_params), std::get<2>(converted_params),           \
                           transform::OpApiParams<decltype(real_params)>(std::move(real_params))); \
  }                                                                                                \
  (#aclnn_api, __VA_ARGS__)

// Async run op.
#define RUN_OP_API(aclnn_api, acl_stream, ...)                \
  do {                                                        \
    transform::RunOpApi(#aclnn_api, acl_stream, __VA_ARGS__); \
  } while (false)

// Sync run op.
#define RUN_OP_API_SYNC(aclnn_api, acl_stream, ...)                                              \
  do {                                                                                           \
    transform::RunOpApi(#aclnn_api, acl_stream, __VA_ARGS__, nullptr);                           \
    auto ret = aclrtSynchronizeStream(acl_stream);                                               \
    if (ret != 0) {                                                                              \
      MS_LOG(EXCEPTION) << "Sync stream " #aclnn_api " failed, detail:" << aclGetRecentErrMsg(); \
    }                                                                                            \
    auto release_mem_func = transform::OpApiDefaultResource::GetInstance().release_mem_func();   \
    if (release_mem_func) {                                                                      \
      release_mem_func(nullptr, false);                                                          \
    }                                                                                            \
    auto uninit_mem_func = transform::OpApiDefaultResource::GetInstance().uninit_mem_func();     \
    if (uninit_mem_func) {                                                                       \
      uninit_mem_func(nullptr, false);                                                           \
    }                                                                                            \
  } while (false)

// Async execute simple micro.
#define EXEC_NPU_CMD(aclnn_api, acl_stream, ...)                                                            \
  do {                                                                                                      \
    static const auto get_workspace_size_func_ptr = transform::GetOpApiFunc(#aclnn_api "GetWorkspaceSize"); \
    if (get_workspace_size_func_ptr == nullptr) {                                                           \
      MS_LOG(EXCEPTION) << #aclnn_api "GetWorkspaceSize"                                                    \
                        << " not in " << transform::GetOpApiLibName() << ", please check!";                 \
    }                                                                                                       \
    static const auto op_api_func = transform::GetOpApiFunc(#aclnn_api);                                    \
    if (op_api_func == nullptr) {                                                                           \
      MS_LOG(EXCEPTION) << #aclnn_api << " not in " << transform::GetOpApiLibName() << ", please check!";   \
    }                                                                                                       \
    uint64_t workspace_size = 0;                                                                            \
    uint64_t *workspace_size_addr = &workspace_size;                                                        \
    transform::aclOpExecutor *executor = nullptr;                                                           \
    transform::aclOpExecutor **executor_addr = &executor;                                                   \
    auto init_mem_func = transform::OpApiDefaultResource::GetInstance().init_mem_func();                    \
    auto uninit_mem_func = transform::OpApiDefaultResource::GetInstance().uninit_mem_func();                \
    if (init_mem_func) {                                                                                    \
      init_mem_func(nullptr, false);                                                                        \
    }                                                                                                       \
    auto converted_params = transform::ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);       \
    static auto get_workspace_size_func =                                                                   \
      transform::ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr);                         \
    auto workspace_status = transform::call(get_workspace_size_func, converted_params);                     \
    if (workspace_status != 0) {                                                                            \
      MS_LOG(EXCEPTION) << #aclnn_api << " not in " << transform::GetOpApiLibName() << ", please check!";   \
    }                                                                                                       \
    void *workspace_addr = nullptr;                                                                         \
    if (workspace_size != 0) {                                                                              \
      workspace_addr = transform::OpApiDefaultResource::GetInstance().AllocWorkspace(workspace_size);       \
    }                                                                                                       \
    auto acl_call = [&converted_params, workspace_addr, &workspace_size, acl_stream, executor]() -> int {   \
      using RunApiFunc = int (*)(void *, uint64_t, transform::aclOpExecutor *, const aclrtStream);          \
      auto run_api_func = reinterpret_cast<RunApiFunc>(op_api_func);                                        \
      auto api_ret = run_api_func(workspace_addr, workspace_size, executor, acl_stream);                    \
      if (api_ret != 0) {                                                                                   \
        MS_LOG(EXCEPTION) << "call " #aclnn_api " failed, detail:" << aclGetRecentErrMsg();                 \
      }                                                                                                     \
      transform::ReleaseConvertTypes(converted_params);                                                     \
      auto release_mem_func = transform::OpApiDefaultResource::GetInstance().release_mem_func();            \
      if (release_mem_func) {                                                                               \
        release_mem_func(nullptr, false);                                                                   \
      }                                                                                                     \
      return api_ret;                                                                                       \
    };                                                                                                      \
    auto call_ret = acl_call();                                                                             \
    if (call_ret != 0) {                                                                                    \
      MS_LOG(EXCEPTION) << #aclnn_api << " exec failed!" << aclGetRecentErrMsg();                           \
    }                                                                                                       \
    if (uninit_mem_func) {                                                                                  \
      uninit_mem_func(nullptr, false);                                                                      \
    }                                                                                                       \
  } while (false)
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_EXEC_H_
